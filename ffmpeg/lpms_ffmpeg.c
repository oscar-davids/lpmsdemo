#include "lpms_ffmpeg.h"

#include <libavcodec/avcodec.h>

#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>

#include <pthread.h>

#ifdef _ADD_LPMS_DNN_
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
#include "tensorflow/c/c_api.h"
#endif

// Not great to appropriate internal API like this...
const int lpms_ERR_INPUT_PIXFMT = FFERRTAG('I','N','P','X');
const int lpms_ERR_FILTERS = FFERRTAG('F','L','T','R');
const int lpms_ERR_PACKET_ONLY = FFERRTAG('P','K','O','N');
const int lpms_ERR_OUTPUTS = FFERRTAG('O','U','T','P');

//
// Internal transcoder data structures
//
struct input_ctx {
  AVFormatContext *ic; // demuxer required
  AVCodecContext  *vc; // video decoder optional
  AVCodecContext  *ac; // audo  decoder optional
  int vi, ai; // video and audio stream indices
  int dv, da; // flags whether to drop video or audio

  // Hardware decoding support
  AVBufferRef *hw_device_ctx;
  enum AVHWDeviceType hw_type;
  char *device;

  int64_t next_pts_a, next_pts_v;

  // Decoder flush
  AVPacket *first_pkt;
  int flushed;
};

struct filter_ctx {
  int active;
  AVFilterGraph *graph;
  AVFrame *frame;
  AVFilterContext *sink_ctx;
  AVFilterContext *src_ctx;

  uint8_t *hwframes; // GPU frame pool data
};

struct output_ctx {
  char *fname;         // required output file name
  char *vfilters;      // required output video filters
  int width, height, bitrate; // w, h, br required
  AVRational fps;
  AVFormatContext *oc; // muxer required
  AVCodecContext  *vc; // video decoder optional
  AVCodecContext  *ac; // audo  decoder optional
  int vi, ai; // video and audio stream indices
  int dv, da; // flags whether to drop video or audio
  struct filter_ctx vf, af;

  // Optional hardware encoding support
  enum AVHWDeviceType hw_type;

  // muxer and encoder information (name + options)
  component_opts *muxer;
  component_opts *video;
  component_opts *audio;

  int64_t drop_ts;     // preroll audio ts to drop

  output_results  *res; // data to return for this output

};

#define MAX_OUTPUT_SIZE 10

struct transcode_thread {
  int initialized;

  struct input_ctx ictx;
  struct output_ctx outputs[MAX_OUTPUT_SIZE];

  int nb_outputs;

};

//some algorithm for yolo 
int max_index(float *a, int n)
{
	if (n <= 0) return -1;
	int i, max_i = 0;
	float max = a[0];
	for (i = 1; i < n; ++i) {
		if (a[i] > max) {
			max = a[i];
			max_i = i;
		}
	}
	return max_i;
}
float overlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

float box_intersection(box a, box b)
{
	float w = overlap(a.x, a.w, b.x, b.w);
	float h = overlap(a.y, a.h, b.y, b.h);
	if (w < 0 || h < 0) return 0;
	float area = w * h;
	return area;
}

float box_union(box a, box b)
{
	float i = box_intersection(a, b);
	float u = a.w*a.h + b.w*b.h - i;
	return u;
}

float box_iou(box a, box b)
{
	//return box_intersection(a, b)/box_union(a, b);

	float I = box_intersection(a, b);
	float U = box_union(a, b);
	if (I == 0 || U == 0) {
		return 0;
	}
	return I / U;
}
float _iou(box box1, box box2)
{
#ifndef ioumax
#define ioumax(a,b) a>b?a:b
#endif
#ifndef ioumin
#define ioumin(a,b) a<b?a:b
#endif
	//Computes Intersection over Union value for 2 bounding boxes
	//param box1 : array of 4 values(top left and bottom right coords) : [x0, y0, x1, y1]
	//param box2 : same as box1

	//float box1.x, box1.y, box1.w, box1.h = box1;
	//float box2.x, box2.y, box2.w, box2.h = box2;

	float int_x0 = ioumax(box1.x, box2.x);
	float int_y0 = ioumax(box1.y, box2.y);
	float int_x1 = ioumin(box1.w, box2.w);
	float int_y1 = ioumin(box1.h, box2.h);

	float int_w = ioumax(int_x1 - int_x0, 0.0);
	float int_h = ioumax(int_y1 - int_y0, 0.0);

	float int_area = int_w * int_h;

	float b1_area = (box1.w - box1.x) * (box1.h - box1.y);
	float b2_area = (box2.w - box2.x) * (box2.h - box2.y);

	//# we add small epsilon of 1e-05 to avoid division by 0
	float iou = int_area / (b1_area + b2_area - int_area + 1e-05);
	return iou;
}
void do_nms(box *boxes, float **probs, int total, int classes, float thresh)
{
	int i, j, k;
	for (i = 0; i < total; ++i) {
		int any = 0;
		for (k = 0; k < classes; ++k) any = any || (probs[i][k] > 0);
		if (!any) {
			continue;
		}
		for (j = i + 1; j < total; ++j) {
			if (boxes[j].w > 0.0 && boxes[j].h > 0.0) {
				if (_iou(boxes[i], boxes[j]) > thresh) {
					for (k = 0; k < classes; ++k) {
						if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
						else probs[j][k] = 0;
					}
				}
			}			
		}
	}
}

int  get_detection_boxes(float* pdata, layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
	int i, j, n;
	float *predictions = pdata;
	int nboxcount = 0;	
	//int per_cell = 5*num+classes;
	for (n = 0; n < l.n; ++n) {
		int index = n;
		//int p_index = l.side*l.side*l.classes + i * l.n + n;
		int p_index = n * l.cols + 4;
		float scale = predictions[p_index];
		if (scale > thresh) {
			nboxcount++;
			
			int box_index = n * l.cols;
			boxes[index].x = (predictions[box_index + 0]);
			boxes[index].y = (predictions[box_index + 1]);			
			boxes[index].w = predictions[box_index + 2];
			boxes[index].h = predictions[box_index + 3];
			for (j = 0; j < l.classes; ++j) {
				int class_index = box_index + 5;
				//float prob = scale * predictions[class_index + j];
				float prob = predictions[class_index + j];
				if (prob > thresh) {
					probs[index][j] = prob;					
				}
				else {
					probs[index][j] = 0.0;
				}
				//probs[index][j] = (prob > thresh) ? prob : 0;
			}
			if (only_objectness) {
				probs[index][0] = scale;
			}
		}
	}
  return nboxcount;	
}
void get_detections(LVPDnnContext *ctx, int nw, int nh, float fx, float fy, int num, float thresh, box *boxes, 
float **probs, int classes, boxobject *objects, int* count)
{
	int i;
	*count = 0;
	int ncount = 0;
  char strtemp[MAXPATH] = {0,};
  if(ctx == NULL || ctx->result == NULL || ctx->resultnum >= MAX_YOLO_FRAME) return;

  //sprintf(ctx->result[ctx->resultnum],"{time:%d,",ctx->resultnum);
  sprintf(ctx->result[ctx->resultnum],"{time:%.3f,",ctx->reftime);
	for (i = 0; i < num; ++i) {
		int class_id = max_index(probs[i], classes);
		float prob = probs[i][class_id];
		if (prob > thresh) {
			
			box b = boxes[i];

			objects[ncount].left = b.x * fx;
			objects[ncount].right = b.w * fx;
			objects[ncount].top = b.y * fy;
			objects[ncount].bot = b.h * fy;

			if (objects[ncount].left < 0) objects[ncount].left = 0;
			if (objects[ncount].right > nw - 1) objects[ncount].right = nw - 1;
			if (objects[ncount].top < 0) objects[ncount].top = 0;
			if (objects[ncount].bot > nh - 1)objects[ncount].bot = nh - 1;
			objects[ncount].prob = prob;
			objects[ncount].class_id = class_id;
      sprintf(strtemp,"%d %.02f %d %d %d %d,", class_id, prob ,objects[ncount].left, objects[ncount].top, objects[ncount].right, objects[ncount].bot);
      if(strlen(ctx->result[ctx->resultnum]) < YOLO_FRESULTMAXPATH - YOLO_FRESULTALLOWPATH){
        strcat(ctx->result[ctx->resultnum], strtemp);
			  ncount++;
      } else {
        break;
      }

		}
	}
	*count = ncount;
  if(ncount > 0) {
    strcat(ctx->result[ctx->resultnum], "}");  
    ctx->resultnum++;
  }
}
//merge transcoding and classify for multi model
DnnFilterNode* Filters = NULL;
int DnnFilterNum = 0;

static int  lpms_detectoneframewithctx(LVPDnnContext *ctx, AVFrame *in);
static int  prepare_sws_context(LVPDnnContext *ctx, AVFrame *frame, int flagHW);

void lpms_setfiltertype(LVPDnnContext* context, int ntype)
{
  if(context == NULL || ntype < 0 || ntype >= (int)DNN_FILTERMAX ) return;
  context->filter_type = (DNNFilterType)ntype;
}
void lpms_dnnCappend(LVPDnnContext* lvpdnn)
{
  DnnFilterNode *t, *temp;
  if(lvpdnn == NULL) return;

  t = (DnnFilterNode*)malloc(sizeof(DnnFilterNode));
  t->data = lvpdnn;
  DnnFilterNum++;

  if (Filters == NULL) {
    Filters = t;
    Filters->next = NULL;    
    return;
  }

  temp = Filters;
  while (temp->next != NULL)
    temp = temp->next;

  temp->next = t;
  t->next   = NULL;
}
void lpms_dnnCdelete(LVPDnnContext* lvpdnn)
{
  DnnFilterNode *t, *tpre, *temp;
  t = tpre = temp = NULL;

  if (Filters == NULL || lvpdnn == NULL) {
    return;
  }

  t = Filters;

  while (t != NULL)  {
    //printf("%d\n", t->data);
    if(lvpdnn == t->data){
      t->data = NULL;
      if(t == Filters){
        temp = Filters->next;
        free(Filters);
        Filters = temp;
        DnnFilterNum--;        
      } else {
        temp = t->next;
        free(t);
        if(tpre)
          tpre->next = temp;
        DnnFilterNum--;        
      }      
      break;

    } else {
      tpre = t;
      t = t->next;
    }    
  }
}
void scanlist() {
  DnnFilterNode *t;
  t = Filters;
  if (t == NULL) {    
    return;
  }
  while (t->next != NULL) {    
    t = t->next;
  }  
}
void initcontextlist() {
  DnnFilterNode *t;
  int ret = 0;
  t = Filters;
  if (t == NULL) {    
    return;
  }
  while (t != NULL) {
    if(t->data->filter_type == DNN_CLASSIFY)
    {
      if(t->data->fmatching && t->data->output.height > 0){
        memset(t->data->fmatching, 0x00, sizeof(float)*t->data->output.height);
      }
    } 
    else if(t->data->filter_type == DNN_YOLO)
    {
      memset(t->data->boxes, 0x00, t->data->output.height*sizeof(box));	    
	    for (int j = 0; j < t->data->output.height; ++j) 
        memset(t->data->probs[j], 0x00, t->data->classes * sizeof(float));

      memset(t->data->object, 0x00, t->data->output.height*sizeof(boxobject));
      //for result      
	    for (int j = 0; j < MAX_YOLO_FRAME; ++j) 
        memset(t->data->result[j], 0x00, MAXPATH);   
    }    
      
    t->data->runcount = 0;
    t->data->resultnum = 0;
    t = t->next;
  }  
}
void cleancontextlist() {
  DnnFilterNode *t;
  int ret = 0;
  t = Filters;
  if (t == NULL) {    
    return;
  }
  while (t != NULL) {
      LVPDnnContext *context = t->data;
      if(context != NULL) {
          if(context->readframe){
              av_frame_free(&context->readframe);
              context->readframe = NULL;
          }
          if(context->swscaleframe){
              av_frame_free(&context->swscaleframe);
              context->swscaleframe = NULL;
          }
          if(context->swframeforHW){
              av_frame_free(&context->swframeforHW);
              context->swframeforHW = NULL;
          }
          if(context->sws_rgb_scale){
            sws_freeContext(context->sws_rgb_scale);
            context->sws_rgb_scale = NULL;
          }
          if(context->sws_gray8_to_grayf32){
            sws_freeContext(context->sws_gray8_to_grayf32);
            context->sws_gray8_to_grayf32 = NULL;
          }
          //release avcontext
          if(context->decoder_ctx){
            avcodec_free_context(&context->decoder_ctx);
            context->decoder_ctx = NULL;
          }
          if(context->input_ctx){
            avformat_close_input(&context->input_ctx);
            context->input_ctx = NULL;
          }  
          if(context->hw_device_ctx){
            context->hw_device_ctx = NULL;
            av_buffer_unref(&context->hw_device_ctx);
          }
          
          context->runcount = 0;
          context->resultnum = 0;
      }
      
      t = t->next;
  }
}

void classifylist(struct AVFrame *frame, int flagHW, float reftime) {
  DnnFilterNode *t;
  int ret = 0;
  t = Filters;
  if (t == NULL) {    
    return;
  }
  while (t != NULL) {
      if(t->data->sws_rgb_scale == NULL || t->data->sws_gray8_to_grayf32 == NULL)
      {
        ret = prepare_sws_context(t->data, frame, flagHW);
        if(ret < 0){
          av_log(NULL, AV_LOG_INFO, "Can not create scale context!\n");
          t = t->next;
          continue;
        }
      }
      //set reftime 
      t->data->reftime = reftime;

      lpms_detectoneframewithctx(t->data,frame);
      t = t->next;
  }
}
void getclassifyresult(int runcount, char* strbuffer) {
  DnnFilterNode *t;
  char stemp[MAXPATH] = {0,};
  int dnnid = 0;
  t = Filters;
  if (t == NULL) {    
    return;
  }
  
  while (t != NULL) {
    //get confidence order 
    if(t->data->filter_type == DNN_CLASSIFY)
    {
      float prob = 0.0;
      int classid = -1; 
      float* confidences = (float*)t->data->fmatching;
      //find max prob classid
      for (int i = 0; i < t->data->output.height; i++)
      {
          if(confidences[i] > prob) {
            prob = confidences[i];
            classid = i;
          }
      }  

      if(runcount > 1) {
        prob = prob / (float)runcount;
      }
      if(prob >= t->data->valid_threshold)
      {
        sprintf(stemp,"%d:%d:%f,", dnnid, classid, prob);
        //strcat
        strcat(strbuffer, stemp);
        av_log(0, AV_LOG_ERROR, "Engine Dnnid Classid & Probability = %d %d %f\n", dnnid, classid, prob);
      }
    } else if(t->data->filter_type == DNN_YOLO) {
      if(t->data->resultnum > 0) {
        //now we print firstframe and last frame object
        sprintf(strbuffer,"%s", t->data->result[0]);
        for (int  i = 1; i < t->data->resultnum; i++)
        {
          if(strlen(strbuffer) < YOLOMAXPATH - MAX_YOLO_FRAME) {
            strcat(strbuffer, t->data->result[i]);
          }
        }
        //for DEBUG    
        av_log(0, AV_LOG_ERROR, "%s\n", strbuffer);
      }

    }
    t = t->next;
    dnnid++;
  }
}
//
// Transcoder
//

output_results * output_results_init(int isYolo)
{
  output_results *result = (output_results*)malloc(sizeof(output_results));
  memset(result, 0x00, sizeof(output_results));
  
  if (isYolo){
    result->desc = (char*)malloc(YOLOMAXPATH * sizeof(char));
    memset(result->desc, 0x00, YOLOMAXPATH * sizeof(char)); 
  } else {
    result->desc = (char*)malloc(MAXPATH * sizeof(char)); 
    memset(result->desc, 0x00, MAXPATH * sizeof(char)); 
  }
  return result;
}
void output_results_destroy(output_results* output_results)
{
  if(output_results == NULL) return;
  if(output_results->desc!= NULL)
      free(output_results->desc);
  free(output_results);
}

static void free_filter(struct filter_ctx *filter)
{
  if (filter->frame) av_frame_free(&filter->frame);
  if (filter->graph) avfilter_graph_free(&filter->graph);
  memset(filter, 0, sizeof(struct filter_ctx));
}

static void close_output(struct output_ctx *octx)
{
  if (octx->oc) {
    if (!(octx->oc->oformat->flags & AVFMT_NOFILE) && octx->oc->pb) {
      avio_closep(&octx->oc->pb);
    }
    avformat_free_context(octx->oc);
    octx->oc = NULL;
  }
  if (octx->vc && AV_HWDEVICE_TYPE_NONE == octx->hw_type) avcodec_free_context(&octx->vc);
  if (octx->ac) avcodec_free_context(&octx->ac);
  free_filter(&octx->vf);
  free_filter(&octx->af);
}

static void free_output(struct output_ctx *octx) {
  close_output(octx);
  if (octx->vc) avcodec_free_context(&octx->vc);
}


static int is_copy(char *encoder) {
  return encoder && !strcmp("copy", encoder);
}

static int is_drop(char *encoder) {
  return !encoder || !strcmp("drop", encoder) || !strcmp("", encoder);
}

static int needs_decoder(char *encoder) {
  // Checks whether the given "encoder" depends on having a decoder.
  // Do this by enumerating special cases that do *not* need encoding
  return !(is_copy(encoder) || is_drop(encoder));
}

static int is_flush_frame(AVFrame *frame)
{
  return -1 == frame->pts;
}

static void send_first_pkt(struct input_ctx *ictx)
{
  if (ictx->flushed || !ictx->first_pkt) return;

  int ret = avcodec_send_packet(ictx->vc, ictx->first_pkt);
  if (ret < 0) {
    char errstr[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, errstr, sizeof errstr);
    fprintf(stderr, "Error sending flush packet : %s\n", errstr);
  }
}

static enum AVPixelFormat hw2pixfmt(AVCodecContext *ctx)
{
  const AVCodec *decoder = ctx->codec;
  struct input_ctx *params = (struct input_ctx*)ctx->opaque;
  for (int i = 0;; i++) {
    const AVCodecHWConfig *config = avcodec_get_hw_config(decoder, i);
    if (!config) {
      fprintf(stderr, "Decoder %s does not support hw decoding\n", decoder->name);
      return AV_PIX_FMT_NONE;
    }
    if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
        config->device_type == params->hw_type) {
      return  config->pix_fmt;
    }
  }
  return AV_PIX_FMT_NONE;
}

static enum AVPixelFormat get_hw_pixfmt(AVCodecContext *vc, const enum AVPixelFormat *pix_fmts)
{
  AVHWFramesContext *frames;
  int ret;

  // XXX Ideally this would be auto initialized by the HW device ctx
  //     However the initialization doesn't occur in time to set up filters
  //     So we do it here. Also see avcodec_get_hw_frames_parameters
  av_buffer_unref(&vc->hw_frames_ctx);
  vc->hw_frames_ctx = av_hwframe_ctx_alloc(vc->hw_device_ctx);
  if (!vc->hw_frames_ctx) {
    fprintf(stderr, "Unable to allocate hwframe context for decoding\n");
    return AV_PIX_FMT_NONE;
  }

  frames = (AVHWFramesContext*)vc->hw_frames_ctx->data;
  frames->format = hw2pixfmt(vc);
  frames->sw_format = vc->sw_pix_fmt;
  frames->width = vc->width;
  frames->height = vc->height;

  // May want to allocate extra HW frames if we encounter samples where
  // the defaults are insufficient. Raising this increases GPU memory usage
  // For now, the defaults seems OK.
  //vc->extra_hw_frames = 16 + 1; // H.264 max refs

  ret = av_hwframe_ctx_init(vc->hw_frames_ctx);
  if (AVERROR(ENOSYS) == ret) ret = lpms_ERR_INPUT_PIXFMT; // most likely
  if (ret < 0) {
    fprintf(stderr,"Unable to initialize a hardware frame pool\n");
    return AV_PIX_FMT_NONE;
  }

/*
fprintf(stderr, "selected format: hw %s sw %s\n",
av_get_pix_fmt_name(frames->format), av_get_pix_fmt_name(frames->sw_format));
const enum AVPixelFormat *p;
for (p = pix_fmts; *p != -1; p++) {
fprintf(stderr,"possible format: %s\n", av_get_pix_fmt_name(*p));
}
*/

  return frames->format;
}

static int init_video_filters(struct input_ctx *ictx, struct output_ctx *octx)
{
#define filters_err(msg) { \
  if (!ret) ret = -1; \
  fprintf(stderr, msg); \
  goto init_video_filters_cleanup; \
}
    char args[512];
    int ret = 0;
    const AVFilter *buffersrc  = avfilter_get_by_name("buffer");
    const AVFilter *buffersink = avfilter_get_by_name("buffersink");
    AVFilterInOut *outputs = NULL;
    AVFilterInOut *inputs  = NULL;
    AVRational time_base = ictx->ic->streams[ictx->vi]->time_base;
    enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_YUV420P, AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE }; // XXX ensure the encoder allows this
    struct filter_ctx *vf = &octx->vf;
    char *filters_descr = octx->vfilters;
    enum AVPixelFormat in_pix_fmt = ictx->vc->pix_fmt;

    // no need for filters with the following conditions
    if (vf->active) goto init_video_filters_cleanup; // already initialized
    if (!needs_decoder(octx->video->name)) goto init_video_filters_cleanup;

    outputs = avfilter_inout_alloc();
    inputs = avfilter_inout_alloc();
    vf->graph = avfilter_graph_alloc();
    if (!outputs || !inputs || !vf->graph) {
      ret = AVERROR(ENOMEM);
      filters_err("Unble to allocate filters\n");
    }
    if (ictx->vc->hw_device_ctx) in_pix_fmt = hw2pixfmt(ictx->vc);

    /* buffer video source: the decoded frames from the decoder will be inserted here. */
    snprintf(args, sizeof args,
            "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
            ictx->vc->width, ictx->vc->height, in_pix_fmt,
            time_base.num, time_base.den,
            ictx->vc->sample_aspect_ratio.num, ictx->vc->sample_aspect_ratio.den);

    ret = avfilter_graph_create_filter(&vf->src_ctx, buffersrc,
                                       "in", args, NULL, vf->graph);
    if (ret < 0) filters_err("Cannot create video buffer source\n");
    if (ictx->vc && ictx->vc->hw_frames_ctx) {
      // XXX a bit problematic in that it's set before decoder is fully ready
      AVBufferSrcParameters *srcpar = av_buffersrc_parameters_alloc();
      srcpar->hw_frames_ctx = ictx->vc->hw_frames_ctx;
      vf->hwframes = ictx->vc->hw_frames_ctx->data;
      av_buffersrc_parameters_set(vf->src_ctx, srcpar);
      av_freep(&srcpar);
    }

    /* buffer video sink: to terminate the filter chain. */
    ret = avfilter_graph_create_filter(&vf->sink_ctx, buffersink,
                                       "out", NULL, NULL, vf->graph);
    if (ret < 0) filters_err("Cannot create video buffer sink\n");

    ret = av_opt_set_int_list(vf->sink_ctx, "pix_fmts", pix_fmts,
                              AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN);
    if (ret < 0) filters_err("Cannot set output pixel format\n");

    /*
     * Set the endpoints for the filter graph. The filter_graph will
     * be linked to the graph described by filters_descr.
     */

    /*
     * The buffer source output must be connected to the input pad of
     * the first filter described by filters_descr; since the first
     * filter input label is not specified, it is set to "in" by
     * default.
     */
    outputs->name       = av_strdup("in");
    outputs->filter_ctx = vf->src_ctx;
    outputs->pad_idx    = 0;
    outputs->next       = NULL;

    /*
     * The buffer sink input must be connected to the output pad of
     * the last filter described by filters_descr; since the last
     * filter output label is not specified, it is set to "out" by
     * default.
     */
    inputs->name       = av_strdup("out");
    inputs->filter_ctx = vf->sink_ctx;
    inputs->pad_idx    = 0;
    inputs->next       = NULL;

    ret = avfilter_graph_parse_ptr(vf->graph, filters_descr,
                                    &inputs, &outputs, NULL);
    if (ret < 0) filters_err("Unable to parse video filters desc\n");

    ret = avfilter_graph_config(vf->graph, NULL);
    if (ret < 0) filters_err("Unable configure video filtergraph\n");

    vf->frame = av_frame_alloc();
    if (!vf->frame) filters_err("Unable to allocate video frame\n");

    vf->active = 1;

init_video_filters_cleanup:
    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);

    return ret;
#undef filters_err
}


static int init_audio_filters(struct input_ctx *ictx, struct output_ctx *octx)
{
#define af_err(msg) { \
  if (!ret) ret = -1; \
  fprintf(stderr, msg); \
  goto init_audio_filters_cleanup; \
}
  int ret = 0;
  char args[512];
  char filters_descr[256];
  const AVFilter *buffersrc  = avfilter_get_by_name("abuffer");
  const AVFilter *buffersink = avfilter_get_by_name("abuffersink");
  AVFilterInOut *outputs = NULL;
  AVFilterInOut *inputs  = NULL;
  struct filter_ctx *af = &octx->af;
  AVRational time_base = ictx->ic->streams[ictx->ai]->time_base;

  // no need for filters with the following conditions
  if (af->active) goto init_audio_filters_cleanup; // already initialized
  if (!needs_decoder(octx->audio->name)) goto init_audio_filters_cleanup;

  outputs = avfilter_inout_alloc();
  inputs = avfilter_inout_alloc();
  af->graph = avfilter_graph_alloc();

  if (!outputs || !inputs || !af->graph) {
    ret = AVERROR(ENOMEM);
    af_err("Unble to allocate audio filters\n");
  }

  /* buffer audio source: the decoded frames from the decoder will be inserted here. */
  snprintf(args, sizeof args,
      "sample_rate=%d:sample_fmt=%d:channel_layout=0x%"PRIx64":channels=%d:"
      "time_base=%d/%d",
      ictx->ac->sample_rate, ictx->ac->sample_fmt, ictx->ac->channel_layout,
      ictx->ac->channels, time_base.num, time_base.den);

  // TODO set sample format and rate based on encoder support,
  //      rather than hardcoding
  snprintf(filters_descr, sizeof filters_descr,
    "aformat=sample_fmts=fltp:channel_layouts=stereo:sample_rates=44100");

  ret = avfilter_graph_create_filter(&af->src_ctx, buffersrc,
                                     "in", args, NULL, af->graph);
  if (ret < 0) af_err("Cannot create audio buffer source\n");

  /* buffer audio sink: to terminate the filter chain. */
  ret = avfilter_graph_create_filter(&af->sink_ctx, buffersink,
                                     "out", NULL, NULL, af->graph);
  if (ret < 0) af_err("Cannot create audio buffer sink\n");

  /*
   * Set the endpoints for the filter graph. The filter_graph will
   * be linked to the graph described by filters_descr.
   */

  /*
   * The buffer source output must be connected to the input pad of
   * the first filter described by filters_descr; since the first
   * filter input label is not specified, it is set to "in" by
   * default.
   */
  outputs->name       = av_strdup("in");
  outputs->filter_ctx = af->src_ctx;
  outputs->pad_idx    = 0;
  outputs->next       = NULL;

  /*
   * The buffer sink input must be connected to the output pad of
   * the last filter described by filters_descr; since the last
   * filter output label is not specified, it is set to "out" by
   * default.
   */
  inputs->name       = av_strdup("out");
  inputs->filter_ctx = af->sink_ctx;
  inputs->pad_idx    = 0;
  inputs->next       = NULL;

  ret = avfilter_graph_parse_ptr(af->graph, filters_descr,
                                &inputs, &outputs, NULL);
  if (ret < 0) af_err("Unable to parse audio filters desc\n");

  ret = avfilter_graph_config(af->graph, NULL);
  if (ret < 0) af_err("Unable configure audio filtergraph\n");

  af->frame = av_frame_alloc();
  if (!af->frame) af_err("Unable to allocate audio frame\n");

  af->active = 1;

init_audio_filters_cleanup:
  avfilter_inout_free(&inputs);
  avfilter_inout_free(&outputs);

  return ret;
#undef af_err
}


static int add_video_stream(struct output_ctx *octx, struct input_ctx *ictx)
{
#define vs_err(msg) { \
  if (!ret) ret = -1; \
  fprintf(stderr, "Error adding video stream: " msg); \
  goto add_video_err; \
}

  // video stream to muxer
  int ret = 0;
  AVStream *st = avformat_new_stream(octx->oc, NULL);
  if (!st) vs_err("Unable to alloc video stream\n");
  octx->vi = st->index;
  st->avg_frame_rate = octx->fps;
  if (is_copy(octx->video->name)) {
    AVStream *ist = ictx->ic->streams[ictx->vi];
    if (ictx->vi < 0 || !ist) vs_err("Input video stream does not exist\n");
    st->time_base = ist->time_base;
    ret = avcodec_parameters_copy(st->codecpar, ist->codecpar);
    if (ret < 0) vs_err("Error copying video params from input stream\n");
    // Sometimes the codec tag is wonky for some reason, so correct it
    ret = av_codec_get_tag2(octx->oc->oformat->codec_tag, st->codecpar->codec_id, &st->codecpar->codec_tag);
    avformat_transfer_internal_stream_timing_info(octx->oc->oformat, st, ist, AVFMT_TBCF_DEMUXER);
  } else if (octx->vc) {
    st->time_base = octx->vc->time_base;
    ret = avcodec_parameters_from_context(st->codecpar, octx->vc);
    if (ret < 0) vs_err("Error setting video params from encoder\n");
  } else vs_err("No video encoder, not a copy; what is this?\n");
  return 0;

add_video_err:
  // XXX free anything here?
  return ret;
#undef vs_err
}

static int add_audio_stream(struct input_ctx *ictx, struct output_ctx *octx)
{
#define as_err(msg) { \
  if (!ret) ret = -1; \
  fprintf(stderr, "Error adding audio stream: " msg); \
  goto add_audio_err; \
}

  if (ictx->ai < 0 || octx->da) {
    // Don't need to add an audio stream if no input audio exists,
    // or we're dropping the output audio stream
    return 0;
  }

  // audio stream to muxer
  int ret = 0;
  AVStream *st = avformat_new_stream(octx->oc, NULL);
  if (!st) as_err("Unable to alloc audio stream\n");
  if (is_copy(octx->audio->name)) {
    AVStream *ist = ictx->ic->streams[ictx->ai];
    if (ictx->ai < 0 || !ist) as_err("Input audio stream does not exist\n");
    st->time_base = ist->time_base;
    ret = avcodec_parameters_copy(st->codecpar, ist->codecpar);
    if (ret < 0) as_err("Error copying audio params from input stream\n");
    // Sometimes the codec tag is wonky for some reason, so correct it
    ret = av_codec_get_tag2(octx->oc->oformat->codec_tag, st->codecpar->codec_id, &st->codecpar->codec_tag);
    avformat_transfer_internal_stream_timing_info(octx->oc->oformat, st, ist, AVFMT_TBCF_DEMUXER);
  } else if (octx->ac) {
    st->time_base = octx->ac->time_base;
    ret = avcodec_parameters_from_context(st->codecpar, octx->ac);
    if (ret < 0) as_err("Error setting audio params from encoder\n");
  } else if (is_drop(octx->audio->name)) {
    // Supposed to exit this function early if there's a drop
    as_err("Shouldn't ever happen here\n");
  } else {
    as_err("No audio encoder; not a copy; what is this?\n");
  }
  octx->ai = st->index;

  // signal whether to drop preroll audio
  if (st->codecpar->initial_padding) octx->drop_ts = AV_NOPTS_VALUE;
  return 0;

add_audio_err:
  // XXX free anything here?
  return ret;
#undef as_err
}

static int open_audio_output(struct input_ctx *ictx, struct output_ctx *octx,
  AVOutputFormat *fmt)
{
#define ao_err(msg) { \
  if (!ret) ret = -1; \
  fprintf(stderr, msg"\n"); \
  goto audio_output_err; \
}

  int ret = 0;
  AVCodec *codec = NULL;
  AVCodecContext *ac = NULL;

  // add audio encoder if a decoder exists and this output requires one
  if (ictx->ac && needs_decoder(octx->audio->name)) {

    // initialize audio filters
    ret = init_audio_filters(ictx, octx);
    if (ret < 0) ao_err("Unable to open audio filter")

    // open encoder
    codec = avcodec_find_encoder_by_name(octx->audio->name);
    if (!codec) ao_err("Unable to find audio encoder");
    // open audio encoder
    ac = avcodec_alloc_context3(codec);
    if (!ac) ao_err("Unable to alloc audio encoder");
    octx->ac = ac;
    ac->sample_fmt = av_buffersink_get_format(octx->af.sink_ctx);
    ac->channel_layout = av_buffersink_get_channel_layout(octx->af.sink_ctx);
    ac->channels = av_buffersink_get_channels(octx->af.sink_ctx);
    ac->sample_rate = av_buffersink_get_sample_rate(octx->af.sink_ctx);
    ac->time_base = av_buffersink_get_time_base(octx->af.sink_ctx);
    if (fmt->flags & AVFMT_GLOBALHEADER) ac->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    ret = avcodec_open2(ac, codec, &octx->audio->opts);
    if (ret < 0) ao_err("Error opening audio encoder");
    av_buffersink_set_frame_size(octx->af.sink_ctx, ac->frame_size);
  }

  ret = add_audio_stream(ictx, octx);
  if (ret < 0) ao_err("Error adding audio stream")

audio_output_err:
  // TODO clean up anything here?
  return ret;

#undef ao_err
}


static int open_output(struct output_ctx *octx, struct input_ctx *ictx, char* metadata)
{
#define em_err(msg) { \
  if (!ret) ret = -1; \
  fprintf(stderr, msg); \
  goto open_output_err; \
}
  int ret = 0, inp_has_stream;

  AVOutputFormat *fmt = NULL;
  AVFormatContext *oc = NULL;
  AVCodecContext *vc  = NULL;
  AVCodec *codec      = NULL;

  // open muxer
  fmt = av_guess_format(octx->muxer->name, octx->fname, NULL);
  if (!fmt) em_err("Unable to guess output format\n");
  ret = avformat_alloc_output_context2(&oc, fmt, NULL, octx->fname);
  if (ret < 0) em_err("Unable to alloc output context\n");
  octx->oc = oc;

  // add video encoder if a decoder exists and this output requires one
  if (ictx->vc && needs_decoder(octx->video->name)) {
    ret = init_video_filters(ictx, octx);
    if (ret < 0) em_err("Unable to open video filter");

    codec = avcodec_find_encoder_by_name(octx->video->name);
    if (!codec) em_err("Unable to find encoder");

    // open video encoder
    // XXX use avoptions rather than manual enumeration
    vc = avcodec_alloc_context3(codec);
    if (!vc) em_err("Unable to alloc video encoder\n");
    octx->vc = vc;
    vc->width = av_buffersink_get_w(octx->vf.sink_ctx);
    vc->height = av_buffersink_get_h(octx->vf.sink_ctx);
    if (octx->fps.den) vc->framerate = av_buffersink_get_frame_rate(octx->vf.sink_ctx);
    else vc->framerate = ictx->vc->framerate;
    if (octx->fps.den) vc->time_base = av_buffersink_get_time_base(octx->vf.sink_ctx);
    else if (ictx->vc->time_base.num && ictx->vc->time_base.den) vc->time_base = ictx->vc->time_base;
    else vc->time_base = ictx->ic->streams[ictx->vi]->time_base;
    if (octx->bitrate) vc->rc_min_rate = vc->rc_max_rate = vc->rc_buffer_size = octx->bitrate;
    if (av_buffersink_get_hw_frames_ctx(octx->vf.sink_ctx)) {
      vc->hw_frames_ctx =
        av_buffer_ref(av_buffersink_get_hw_frames_ctx(octx->vf.sink_ctx));
      if (!vc->hw_frames_ctx) em_err("Unable to alloc hardware context\n");
    }
    vc->pix_fmt = av_buffersink_get_format(octx->vf.sink_ctx); // XXX select based on encoder + input support
    if (fmt->flags & AVFMT_GLOBALHEADER) vc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    ret = avcodec_open2(vc, codec, &octx->video->opts);
    if (ret < 0) em_err("Error opening video encoder\n");
    octx->hw_type = ictx->hw_type;
  }

  // add video stream if input contains video
  inp_has_stream = ictx->vi >= 0;
  if (inp_has_stream && !octx->dv) {
    ret = add_video_stream(octx, ictx);
    if (ret < 0) em_err("Error adding video stream\n");
  }

  ret = open_audio_output(ictx, octx, fmt);
  if (ret < 0) em_err("Error opening audio output\n");

  if (!(fmt->flags & AVFMT_NOFILE)) {
    ret = avio_open(&octx->oc->pb, octx->fname, AVIO_FLAG_WRITE);
    if (ret < 0) em_err("Error opening output file\n");
  }
  if(metadata != NULL && strlen(metadata) > 0)
  {
    AVDictionary *pmetadata = NULL;
    av_dict_set(&pmetadata, "title", metadata, 0);
    oc->metadata = pmetadata;
    //for debug metadata
    //av_log(0, AV_LOG_ERROR, "Engine metadata = %s\n", metadata);
  } 

  ret = avformat_write_header(oc, &octx->muxer->opts);
  if (ret < 0) em_err("Error writing header\n");

  return 0;

open_output_err:
  free_output(octx);
  return ret;
}

static void free_input(struct input_ctx *inctx)
{
  if (inctx->ic) avformat_close_input(&inctx->ic);
  if (inctx->vc) {
    if (inctx->vc->hw_device_ctx) av_buffer_unref(&inctx->vc->hw_device_ctx);
    avcodec_free_context(&inctx->vc);
  }
  if (inctx->ac) avcodec_free_context(&inctx->ac);
  if (inctx->hw_device_ctx) av_buffer_unref(&inctx->hw_device_ctx);
}

static int open_video_decoder(input_params *params, struct input_ctx *ctx)
{
#define dd_err(msg) { \
  if (!ret) ret = -1; \
  fprintf(stderr, msg); \
  goto open_decoder_err; \
}
  int ret = 0;
  AVCodec *codec = NULL;
  AVFormatContext *ic = ctx->ic;

  // open video decoder
  ctx->vi = av_find_best_stream(ic, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
  if (ctx->dv) ; // skip decoding video
  else if (ctx->vi < 0) {
    fprintf(stderr, "No video stream found in input\n");
  } else {
    if (AV_CODEC_ID_H264 == codec->id &&
        AV_HWDEVICE_TYPE_CUDA == params->hw_type) {
      AVCodec *c = avcodec_find_decoder_by_name("h264_cuvid");
      if (c) codec = c;
      else fprintf(stderr, "Cuvid decoder not found; defaulting to software\n");
      if (AV_PIX_FMT_YUV420P != ic->streams[ctx->vi]->codecpar->format) {
        ret = lpms_ERR_INPUT_PIXFMT;
        dd_err("Non 4:2:0 pixel format detected in input\n");
      }
    }
    AVCodecContext *vc = avcodec_alloc_context3(codec);
    if (!vc) dd_err("Unable to alloc video codec\n");
    ctx->vc = vc;
    ret = avcodec_parameters_to_context(vc, ic->streams[ctx->vi]->codecpar);
    if (ret < 0) dd_err("Unable to assign video params\n");
    vc->opaque = (void*)ctx;
    // XXX Could this break if the original device falls out of scope in golang?
    if (params->hw_type != AV_HWDEVICE_TYPE_NONE) {
      // First set the hw device then set the hw frame
      ret = av_hwdevice_ctx_create(&ctx->hw_device_ctx, params->hw_type, params->device, NULL, 0);
      if (ret < 0) dd_err("Unable to open hardware context for decoding\n")
      ctx->hw_type = params->hw_type;
      vc->hw_device_ctx = av_buffer_ref(ctx->hw_device_ctx);
      vc->get_format = get_hw_pixfmt;
    }
    vc->pkt_timebase = ic->streams[ctx->vi]->time_base;
    ret = avcodec_open2(vc, codec, NULL);
    if (ret < 0) dd_err("Unable to open video decoder\n");
  }

  return 0;

open_decoder_err:
  free_input(ctx);
  return ret;
#undef dd_err
}

static int open_audio_decoder(input_params *params, struct input_ctx *ctx)
{
#define ad_err(msg) { \
  if (!ret) ret = -1; \
  fprintf(stderr, msg); \
  goto open_audio_err; \
}
  int ret = 0;
  AVCodec *codec = NULL;
  AVFormatContext *ic = ctx->ic;

  // open audio decoder
  ctx->ai = av_find_best_stream(ic, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
  if (ctx->da) ; // skip decoding audio
  else if (ctx->ai < 0) {
    fprintf(stderr, "No audio stream found in input\n");
  } else {
    AVCodecContext * ac = avcodec_alloc_context3(codec);
    if (!ac) ad_err("Unable to alloc audio codec\n");
    if (ctx->ac) fprintf(stderr, "Audio context already open! %p\n", ctx->ac);
    ctx->ac = ac;
    ret = avcodec_parameters_to_context(ac, ic->streams[ctx->ai]->codecpar);
    if (ret < 0) ad_err("Unable to assign audio params\n");
    ret = avcodec_open2(ac, codec, NULL);
    if (ret < 0) ad_err("Unable to open audio decoder\n");
  }

  return 0;

open_audio_err:
  free_input(ctx);
  return ret;
#undef ad_err
}

static int open_input(input_params *params, struct input_ctx *ctx)
{
#define dd_err(msg) { \
  if (!ret) ret = -1; \
  fprintf(stderr, msg); \
  goto open_input_err; \
}
  AVFormatContext *ic   = NULL;
  AVIOContext *pb       = NULL;
  char *inp = params->fname;
  int ret = 0;

  // open demuxer
  ic = avformat_alloc_context();
  if (!ic) dd_err("demuxer: Unable to alloc context\n");
  ret = avio_open(&pb, inp, AVIO_FLAG_READ);
  if (ret < 0) dd_err("demuxer: Unable to open file\n");
  ic->pb = pb;
  ret = avformat_open_input(&ic, NULL, NULL, NULL);
  if (ret < 0) dd_err("demuxer: Unable to open input\n");
  ctx->ic = ic;
  ret = avformat_find_stream_info(ic, NULL);
  if (ret < 0) dd_err("Unable to find input info\n");
  ret = open_video_decoder(params, ctx);
  if (ret < 0) dd_err("Unable to open video decoder\n")
  ret = open_audio_decoder(params, ctx);
  if (ret < 0) dd_err("Unable to open audio decoder\n")

  return 0;

open_input_err:
  fprintf(stderr, "Freeing input based on OPEN INPUT error\n");
  avio_close(pb); // need to close manually, avformat_open_input
                  // not closes it in case of error
  free_input(ctx);
  return ret;
#undef dd_err
}

int process_in(struct input_ctx *ictx, AVFrame *frame, AVPacket *pkt)
{
#define dec_err(msg) { \
  if (!ret) ret = -1; \
  fprintf(stderr, msg); \
  goto dec_cleanup; \
}
  int ret = 0;

  // Read a packet and attempt to decode it.
  // If decoding was not possible, return the packet anyway for streamcopy
  av_init_packet(pkt);
  // TODO this while-loop isn't necessary anymore; clean up
  while (1) {
    AVStream *ist = NULL;
    AVCodecContext *decoder = NULL;
    ret = av_read_frame(ictx->ic, pkt);
    if (ret == AVERROR_EOF) goto dec_flush;
    else if (ret < 0) dec_err("Unable to read input\n");
    ist = ictx->ic->streams[pkt->stream_index];
    if (ist->index == ictx->vi && ictx->vc) decoder = ictx->vc;
    else if (ist->index == ictx->ai && ictx->ac) decoder = ictx->ac;
    else if (pkt->stream_index == ictx->vi || pkt->stream_index == ictx->ai) break;
    else dec_err("Could not find decoder or stream\n");

    if (!ictx->first_pkt && pkt->flags & AV_PKT_FLAG_KEY && decoder == ictx->vc) {
      ictx->first_pkt = av_packet_clone(pkt);
      ictx->first_pkt->pts = -1;
    }

    ret = avcodec_send_packet(decoder, pkt);
    if (ret < 0) dec_err("Error sending packet to decoder\n");
    ret = avcodec_receive_frame(decoder, frame);
    if (ret == AVERROR(EAGAIN)) {
      // Distinguish from EAGAIN that may occur with
      // av_read_frame or avcodec_send_packet
      ret = lpms_ERR_PACKET_ONLY;
      break;
    }
    else if (ret < 0) dec_err("Error receiving frame from decoder\n");
    break;
  }

dec_cleanup:
  return ret;

dec_flush:

  // Attempt to read all frames that are remaining within the decoder, starting
  // with video. If there's a nonzero response type, we know there are no more
  // video frames, so continue on to audio.

  // Flush video decoder.
  // To accommodate CUDA, we feed the decoder a a sentinel (flush) frame.
  // Once the flush frame has been decoded, the decoder is fully flushed.
  // TODO this is unnecessary for SW decoding! SW process should match audio
  if (ictx->vc) {
    send_first_pkt(ictx);

    ret = avcodec_receive_frame(ictx->vc, frame);
    pkt->stream_index = ictx->vi;
    if (!ret) {
      if (is_flush_frame(frame)) ictx->flushed = 1;
      return ret;
    }
  }
  // Flush audio decoder.
  if (ictx->ac) {
    avcodec_send_packet(ictx->ac, NULL);
    ret = avcodec_receive_frame(ictx->ac, frame);
    pkt->stream_index = ictx->ai;
    if (!ret) return ret;
  }
  return AVERROR_EOF;

#undef dec_err
}

static int mux(AVPacket *pkt, AVRational tb, struct output_ctx *octx, AVStream *ost)
{
  pkt->stream_index = ost->index;
  if (av_cmp_q(tb, ost->time_base)) {
    av_packet_rescale_ts(pkt, tb, ost->time_base);
  }

  // drop any preroll audio. may need to drop multiple packets for multichannel
  // XXX this breaks if preroll isn't exactly one AVPacket or drop_ts == 0
  //     hasn't been a problem in practice (so far)
  if (AVMEDIA_TYPE_AUDIO == ost->codecpar->codec_type) {
      if (octx->drop_ts == AV_NOPTS_VALUE) octx->drop_ts = pkt->pts;
      if (pkt->pts && pkt->pts == octx->drop_ts) return 0;
  }

  return av_interleaved_write_frame(octx->oc, pkt);
}

int encode(AVCodecContext* encoder, AVFrame *frame, struct output_ctx* octx, AVStream* ost) {
#define encode_err(msg) { \
  char errstr[AV_ERROR_MAX_STRING_SIZE] = {0}; \
  if (!ret) { fprintf(stderr, "should not happen\n"); ret = AVERROR(ENOMEM); } \
  if (ret < -1) av_strerror(ret, errstr, sizeof errstr); \
  fprintf(stderr, "%s: %s\n", msg, errstr); \
  goto encode_cleanup; \
}

  int ret = 0;
  AVPacket pkt = {0};

  if (AVMEDIA_TYPE_VIDEO == ost->codecpar->codec_type && frame) {
    if (!octx->res->frames) {
      frame->pict_type = AV_PICTURE_TYPE_I;
    }
    octx->res->frames++;
    octx->res->pixels += encoder->width * encoder->height;
  }


  // We don't want to send NULL frames for HW encoding
  // because that closes the encoder: not something we want
  if (AV_HWDEVICE_TYPE_NONE == octx->hw_type || frame) {
    ret = avcodec_send_frame(encoder, frame);
    if (AVERROR_EOF == ret) ; // continue ; drain encoder
    else if (ret < 0) encode_err("Error sending frame to encoder");
  }

  if (AVMEDIA_TYPE_VIDEO == ost->codecpar->codec_type &&
      AV_HWDEVICE_TYPE_CUDA == octx->hw_type && !frame) {
    avcodec_flush_buffers(encoder);
  }

  while (1) {
    av_init_packet(&pkt);
    ret = avcodec_receive_packet(encoder, &pkt);
    if (AVERROR(EAGAIN) == ret || AVERROR_EOF == ret) goto encode_cleanup;
    if (ret < 0) encode_err("Error receiving packet from encoder\n");
    ret = mux(&pkt, encoder->time_base, octx, ost);
    if (ret < 0) goto encode_cleanup;
    av_packet_unref(&pkt);
  }

encode_cleanup:
  av_packet_unref(&pkt);
  return ret;

#undef encode_err
}

int process_out(struct input_ctx *ictx, struct output_ctx *octx, AVCodecContext *encoder, AVStream *ost,
  struct filter_ctx *filter, AVFrame *inf)
{
#define proc_err(msg) { \
  char errstr[AV_ERROR_MAX_STRING_SIZE] = {0}; \
  if (!ret) { fprintf(stderr, "u done messed up\n"); ret = AVERROR(ENOMEM); } \
  if (ret < -1) av_strerror(ret, errstr, sizeof errstr); \
  fprintf(stderr, "%s: %s\n", msg, errstr); \
  goto proc_cleanup; \
}
  int ret = 0;

  if (!encoder) proc_err("Trying to transmux; not supported")

  if (!filter || !filter->active) {
    // No filter in between decoder and encoder, so use input frame directly
    return encode(encoder, inf, octx, ost);
  }

  // Sometimes we have to reset the filter if the HW context is updated
  // because we initially set the filter before the decoder is fully ready
  // and the decoder may change HW params
  if (AVMEDIA_TYPE_VIDEO == ost->codecpar->codec_type &&
      inf && inf->hw_frames_ctx && filter->hwframes &&
      inf->hw_frames_ctx->data != filter->hwframes) {
    free_filter(&octx->vf); // XXX really should flush filter first
    ret = init_video_filters(ictx, octx);
    if (ret < 0) return lpms_ERR_FILTERS;
  }
  if (inf) {
    ret = av_buffersrc_write_frame(filter->src_ctx, inf);
    if (ret < 0) proc_err("Error feeding the filtergraph");
  } else {
    // We need to set the pts at EOF to the *end* of the last packet
    // in order to avoid discarding any queued packets
    int64_t next_pts = AVMEDIA_TYPE_VIDEO == ost->codecpar->codec_type ?
      ictx->next_pts_v : ictx->next_pts_a;
    av_buffersrc_close(filter->src_ctx, next_pts, AV_BUFFERSRC_FLAG_PUSH);
  }

  while (1) {
    // Drain the filter. Each input frame may have multiple output frames
    AVFrame *frame = filter->frame;
    av_frame_unref(frame);
    ret = av_buffersink_get_frame(filter->sink_ctx, frame);
    frame->pict_type = AV_PICTURE_TYPE_NONE;
    if (AVERROR(EAGAIN) == ret || AVERROR_EOF == ret) {
      // no frame returned from filtergraph
      // proceed only if the input frame is a flush (inf == null)
      if (inf) return ret;
      frame = NULL;
    } else if (ret < 0) proc_err("Error consuming the filtergraph\n");
    ret = encode(encoder, frame, octx, ost);
    av_frame_unref(frame);
    // For HW we keep the encoder open so will only get EAGAIN.
    // Return EOF in place of EAGAIN for to terminate the flush
    if (frame == NULL && AV_HWDEVICE_TYPE_NONE != octx->hw_type &&
        AVERROR(EAGAIN) == ret && !inf) return AVERROR_EOF;
    if (frame == NULL) return ret;
  }

proc_cleanup:
  return ret;
#undef proc_err
}

int flush_outputs(struct input_ctx *ictx, struct output_ctx *octx)
{
  // only issue w this flushing method is it's not necessarily sequential
  // wrt all the outputs; might want to iterate on each output per frame?
  int ret = 0;
  if (octx->vc) { // flush video
    while (!ret || ret == AVERROR(EAGAIN)) {
      ret = process_out(ictx, octx, octx->vc, octx->oc->streams[octx->vi], &octx->vf, NULL);
    }
  }
  ret = 0;
  if (octx->ac) { // flush audio
    while (!ret || ret == AVERROR(EAGAIN)) {
      ret = process_out(ictx, octx, octx->ac, octx->oc->streams[octx->ai], &octx->af, NULL);
    }
  }
  av_interleaved_write_frame(octx->oc, NULL); // flush muxer
  return av_write_trailer(octx->oc);
}
//
// Segmenter
//

int lpms_rtmp2hls(char *listen, char *outf, char *ts_tmpl, char* seg_time, char *seg_start)
{
#define r2h_err(str) {\
  if (!ret) ret = 1; \
  errstr = str; \
  goto handle_r2h_err; \
}
  char *errstr          = NULL;
  int ret               = 0;
  AVFormatContext *ic   = NULL;
  AVFormatContext *oc   = NULL;
  AVOutputFormat *ofmt  = NULL;
  AVStream *ist         = NULL;
  AVStream *ost         = NULL;
  AVDictionary *md      = NULL;
  AVCodec *codec        = NULL;
  int64_t prev_ts[2]    = {AV_NOPTS_VALUE, AV_NOPTS_VALUE};
  int stream_map[2]     = {-1, -1};
  int got_video_kf      = 0;
  AVPacket pkt;

  ret = avformat_open_input(&ic, listen, NULL, NULL);
  if (ret < 0) r2h_err("segmenter: Unable to open input\n");
  ret = avformat_find_stream_info(ic, NULL);
  if (ret < 0) r2h_err("segmenter: Unable to find any input streams\n");

  ofmt = av_guess_format(NULL, outf, NULL);
  if (!ofmt) r2h_err("Could not deduce output format from file extension\n");
  ret = avformat_alloc_output_context2(&oc, ofmt, NULL, outf);
  if (ret < 0) r2h_err("Unable to allocate output context\n");

  // XXX accommodate cases where audio or video is empty
  stream_map[0] = av_find_best_stream(ic, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
  if (stream_map[0] < 0) r2h_err("segmenter: Unable to find video stream\n");
  stream_map[1] = av_find_best_stream(ic, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
  if (stream_map[1] < 0) r2h_err("segmenter: Unable to find audio stream\n");

  ist = ic->streams[stream_map[0]];
  ost = avformat_new_stream(oc, NULL);
  if (!ost) r2h_err("segmenter: Unable to allocate output video stream\n");
  avcodec_parameters_copy(ost->codecpar, ist->codecpar);
  ist = ic->streams[stream_map[1]];
  ost = avformat_new_stream(oc, NULL);
  if (!ost) r2h_err("segmenter: Unable to allocate output audio stream\n");
  avcodec_parameters_copy(ost->codecpar, ist->codecpar);

  av_dict_set(&md, "hls_time", seg_time, 0);
  av_dict_set(&md, "hls_segment_filename", ts_tmpl, 0);
  av_dict_set(&md, "start_number", seg_start, 0);
  av_dict_set(&md, "hls_flags", "delete_segments", 0);
  ret = avformat_write_header(oc, &md);
  if (ret < 0) r2h_err("Error writing header\n");

  av_init_packet(&pkt);
  while (1) {
    ret = av_read_frame(ic, &pkt);
    if (ret == AVERROR_EOF) {
      av_interleaved_write_frame(oc, NULL); // flush
      break;
    } else if (ret < 0) r2h_err("Error reading\n");
    // rescale timestamps
    if (pkt.stream_index == stream_map[0]) pkt.stream_index = 0;
    else if (pkt.stream_index == stream_map[1]) pkt.stream_index = 1;
    else goto r2hloop_end;
    ist = ic->streams[stream_map[pkt.stream_index]];
    ost = oc->streams[pkt.stream_index];
    int64_t dts_next = pkt.dts, dts_prev = prev_ts[pkt.stream_index];
    if (oc->streams[pkt.stream_index]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO &&
        AV_NOPTS_VALUE == dts_prev &&
        (pkt.flags & AV_PKT_FLAG_KEY)) got_video_kf = 1;
    if (!got_video_kf) goto r2hloop_end; // skip everyting until first video KF
    if (AV_NOPTS_VALUE == dts_prev) dts_prev = dts_next;
    else if (dts_next <= dts_prev) goto r2hloop_end; // drop late packets
    pkt.pts = av_rescale_q_rnd(pkt.pts, ist->time_base, ost->time_base,
        AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
    pkt.dts = av_rescale_q_rnd(pkt.dts, ist->time_base, ost->time_base,
        AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
    if (!pkt.duration) pkt.duration = dts_next - dts_prev;
    pkt.duration = av_rescale_q(pkt.duration, ist->time_base, ost->time_base);
    prev_ts[pkt.stream_index] = dts_next;
    // write the thing
    ret = av_interleaved_write_frame(oc, &pkt);
    if (ret < 0) r2h_err("segmenter: Unable to write output frame\n");
r2hloop_end:
    av_packet_unref(&pkt);
  }
  ret = av_write_trailer(oc);
  if (ret < 0) r2h_err("segmenter: Unable to write trailer\n");

handle_r2h_err:
  if (errstr) fprintf(stderr, "%s", errstr);
  if (ic) avformat_close_input(&ic);
  if (oc) avformat_free_context(oc);
  if (md) av_dict_free(&md);
  return ret == AVERROR_EOF ? 0 : ret;
}


int transcode(struct transcode_thread *h,
  input_params *inp, output_params *params,
  output_results *results, output_results *decoded_results)
{
#define main_err(msg) { \
  char errstr[AV_ERROR_MAX_STRING_SIZE] = {0}; \
  if (!ret) ret = AVERROR(EINVAL); \
  if (ret < -1) av_strerror(ret, errstr, sizeof errstr); \
  fprintf(stderr, "%s: %s\n", msg, errstr); \
  goto transcode_cleanup; \
}
  int ret = 0, i = 0;
  struct input_ctx *ictx = &h->ictx;
  struct output_ctx *outputs = h->outputs;
  int nb_outputs = h->nb_outputs;
  AVPacket ipkt;
  AVFrame *dframe = NULL;
  //added module for classify
  int runclassify = 0;
  int nsamplerate = 0;
  float framtime = 0.0; //unit second
  int flagHW = AV_HWDEVICE_TYPE_CUDA == ictx->hw_type;
  if(inp->ftimeinterval > 0.0) nsamplerate = (int)(25.0 * inp->ftimeinterval);
  else {
    if(Filters != NULL)
      nsamplerate = Filters->data->sample_rate;    
  }
  initcontextlist();
  //av_log(0, AV_LOG_ERROR, "nsamplerate = %d \n",nsamplerate);

  if (!inp) main_err("transcoder: Missing input params\n")

  if (!ictx->ic->pb) {
    ret = avio_open(&ictx->ic->pb, inp->fname, AVIO_FLAG_READ);
    if (ret < 0) main_err("Unable to reopen file");
    // XXX check to see if we can also reuse decoder for sw decoding
    if (AV_HWDEVICE_TYPE_CUDA != ictx->hw_type) {
      ret = open_video_decoder(inp, ictx);
      if (ret < 0) main_err("Unable to reopen video decoder");
    }
    ret = open_audio_decoder(inp, ictx);
    if (ret < 0) main_err("Unable to reopen audio decoder")
  }

  // populate output contexts
  for (i = 0; i <  nb_outputs; i++) {
      struct output_ctx *octx = &outputs[i];
      octx->fname = params[i].fname;
      octx->width = params[i].w;
      octx->height = params[i].h;
      octx->muxer = &params[i].muxer;
      octx->audio = &params[i].audio;
      octx->video = &params[i].video;
      octx->vfilters = params[i].vfilters;
      if (params[i].bitrate) octx->bitrate = params[i].bitrate;
      if (params[i].fps.den) octx->fps = params[i].fps;
      octx->dv = ictx->vi < 0 || is_drop(octx->video->name);
      octx->da = ictx->ai < 0 || is_drop(octx->audio->name);
      octx->res = &results[i];

      // XXX valgrind this line up
      if (!h->initialized || AV_HWDEVICE_TYPE_NONE == octx->hw_type) {
        ret = open_output(octx, ictx, inp->metadata);
        if (ret < 0) main_err("transcoder: Unable to open output");        
        continue;
      }

      // reopen output for HW encoding

      AVOutputFormat *fmt = av_guess_format(octx->muxer->name, octx->fname, NULL);
      if (!fmt) main_err("Unable to guess format for reopen\n");
      ret = avformat_alloc_output_context2(&octx->oc, fmt, NULL, octx->fname);
      if (ret < 0) main_err("Unable to alloc reopened out context\n");

      // re-attach video encoder
      if (octx->vc) {
        ret = add_video_stream(octx, ictx);
        if (ret < 0) main_err("Unable to re-add video stream\n");
        ret = init_video_filters(ictx, octx);
        if (ret < 0) main_err("Unable to re-open video filter\n")
      } else fprintf(stderr, "no video stream\n");

      // re-attach audio encoder
      ret = open_audio_output(ictx, octx, fmt);
      if (ret < 0) main_err("Unable to re-add audio stream\n");

      if (!(fmt->flags & AVFMT_NOFILE)) {
        ret = avio_open(&octx->oc->pb, octx->fname, AVIO_FLAG_WRITE);
        if (ret < 0) main_err("Error re-opening output file\n");
      }
      //write metadata      
      if(inp->metadata != NULL && strlen(inp->metadata) > 0)
      {
        AVDictionary *pmetadata = NULL;
        av_dict_set(&pmetadata, "title", inp->metadata, 0);
        octx->oc->metadata = pmetadata;
        //for debug
        //av_log(0, AV_LOG_ERROR, "Engine metadata = %s\n",inp->metadata);
      } 

      ret = avformat_write_header(octx->oc, NULL);
      if (ret < 0) main_err("Error re-writing header\n");
  }

  av_init_packet(&ipkt);
  dframe = av_frame_alloc();
  if (!dframe) main_err("transcoder: Unable to allocate frame\n");

  while (1) {
    int has_frame = 0;
    AVStream *ist = NULL;
    av_frame_unref(dframe);
    ret = process_in(ictx, dframe, &ipkt);
    if (ret == AVERROR_EOF) break;
                            // Bail out on streams that appear to be broken
    else if (lpms_ERR_PACKET_ONLY == ret) ; // keep going for stream copy
    else if (ret < 0) main_err("transcoder: Could not decode; stopping\n");
    ist = ictx->ic->streams[ipkt.stream_index];
    has_frame = lpms_ERR_PACKET_ONLY != ret;

    if (AVMEDIA_TYPE_VIDEO == ist->codecpar->codec_type) {
      if (is_flush_frame(dframe)) goto whileloop_end;
      // width / height will be zero for pure streamcopy (no decoding)
      decoded_results->frames += dframe->width && dframe->height;
      decoded_results->pixels += dframe->width * dframe->height;
      if (has_frame) {
        int64_t dur = 0;
        if (dframe->pkt_duration) dur = dframe->pkt_duration;
        else if (ist->avg_frame_rate.den) {
          dur = av_rescale_q(1, av_inv_q(ist->avg_frame_rate), ist->time_base);
        } else {
          // TODO use better heuristics for this; look at how ffmpeg does it
          //fprintf(stderr, "Could not determine next pts; filter might drop\n");
        }
        ictx->next_pts_v = dframe->pts + dur;
        //invoke call classification
        if(DnnFilterNum > 0 && nsamplerate > 0 && (decoded_results->frames - 1) % nsamplerate == 0){
          runclassify++;
          float sampleinterval = 1.0 / 25.0;
        
          if(dframe->pts == AV_NOPTS_VALUE){
            framtime = 0.0;
          } else {

              if(ist->r_frame_rate.den > 0.0){
                sampleinterval = 1.0 / av_q2d(ist->r_frame_rate);
              } else  {
                sampleinterval = av_q2d(ist->time_base);    
              }
              
            framtime = (decoded_results->frames - 1) * sampleinterval;
          }          
          //for DEBUG
          //av_log(0, AV_LOG_ERROR, "DnnFilter frame time = %f\n",framtime);
          //av_log(0, AV_LOG_INFO, "DnnFilterNum num frame nsamplerate = %d %d\n",DnnFilterNum,decoded_results->frames,nsamplerate);
          classifylist(dframe, flagHW, framtime);          
        }
      }
    } else if (AVMEDIA_TYPE_AUDIO == ist->codecpar->codec_type) {
      if (has_frame) ictx->next_pts_a = dframe->pts + dframe->pkt_duration;
    }

    for (i = 0; i < nb_outputs; i++) {
      struct output_ctx *octx = &outputs[i];
      struct filter_ctx *filter = NULL;
      AVStream *ost = NULL;
      AVCodecContext *encoder = NULL;
      ret = 0; // reset to avoid any carry-through

      if (ist->index == ictx->vi) {
        if (octx->dv) continue; // drop video stream for this output
        ost = octx->oc->streams[0];
        if (ictx->vc) {
          encoder = octx->vc;
          filter = &octx->vf;
        }
      } else if (ist->index == ictx->ai) {
        if (octx->da) continue; // drop audio stream for this output
        ost = octx->oc->streams[!octx->dv]; // depends on whether video exists
        if (ictx->ac) {
          encoder = octx->ac;
          filter = &octx->af;
        }
      } else continue; // dropped or unrecognized stream

      if (!encoder && ost) {
        // stream copy
        AVPacket *pkt;

        // we hit this case when decoder is flushing; will be no input packet
        // (we don't need decoded frames since this stream is doing a copy)
        if (ipkt.pts == AV_NOPTS_VALUE) continue;

        pkt = av_packet_clone(&ipkt);
        if (!pkt) main_err("transcoder: Error allocating packet\n");
        ret = mux(pkt, ist->time_base, octx, ost);
        av_packet_free(&pkt);		
      } else if (has_frame) {
        ret = process_out(ictx, octx, encoder, ost, filter, dframe);
      }
      if (AVERROR(EAGAIN) == ret || AVERROR_EOF == ret) continue;
      else if (ret < 0) main_err("transcoder: Error encoding\n");
    }
whileloop_end:
    av_packet_unref(&ipkt);
  }

  // flush outputs
  for (i = 0; i < nb_outputs; i++) {
    ret = flush_outputs(ictx, &outputs[i]);
    if (ret < 0) main_err("transcoder: Unable to fully flush outputs")
  }
  //make classification result
  if(DnnFilterNum > 0){
    getclassifyresult(runclassify,decoded_results->desc);
    cleancontextlist();
  }

transcode_cleanup:
  avio_closep(&ictx->ic->pb);
  if (dframe) av_frame_free(&dframe);
  ictx->flushed = 0;
  if (ictx->first_pkt) av_packet_free(&ictx->first_pkt);
  if (ictx->ac) avcodec_free_context(&ictx->ac);
  if (ictx->vc && AV_HWDEVICE_TYPE_NONE == ictx->hw_type) avcodec_free_context(&ictx->vc);
  for (i = 0; i < nb_outputs; i++) close_output(&outputs[i]);
  return ret == AVERROR_EOF ? 0 : ret;
#undef main_err
}

void lpms_init()
{
  av_log_set_level(AV_LOG_INFO);
}
int lpms_transcode(input_params *inp, output_params *params,
  output_results *results, int nb_outputs, output_results *decoded_results)
{
  int ret = 0;
  struct transcode_thread *h = inp->handle;

  if (!h->initialized) {
    int i = 0;
    int decode_a = 0, decode_v = 0;
    if (nb_outputs > MAX_OUTPUT_SIZE) {
      return lpms_ERR_OUTPUTS;
    }

    // Check to see if we can skip decoding
    for (i = 0; i < nb_outputs; i++) {
      if (!needs_decoder(params[i].video.name)) h->ictx.dv = ++decode_v == nb_outputs;
      if (!needs_decoder(params[i].audio.name)) h->ictx.da = ++decode_a == nb_outputs;
    }

    h->nb_outputs = nb_outputs;

    // populate input context
    ret = open_input(inp, &h->ictx);
    if (ret < 0) {
      return ret;
    }
  }

  if (h->nb_outputs != nb_outputs) {
    return lpms_ERR_OUTPUTS; // Not the most accurate error...
  }

  ret = transcode(h, inp, params, results, decoded_results);
  h->initialized = 1;
  return ret;
}

struct transcode_thread* lpms_transcode_new() {
  struct transcode_thread *h = malloc(sizeof (struct transcode_thread));
  if (!h) return NULL;
  memset(h, 0, sizeof *h);
  return h;
}

void lpms_transcode_stop(struct transcode_thread *handle) {
  // not threadsafe as-is; calling function must ensure exclusivity!

  int i;

  if (!handle) return;

  free_input(&handle->ictx);
  for (i = 0; i < MAX_OUTPUT_SIZE; i++) {
    free_output(&handle->outputs[i]);
  }

  free(handle);
}

#ifdef _ADD_LPMS_DNN_

DNNModel *ff_dnn_load_model_tf(const char *model_filename);

DNNReturnType ff_dnn_execute_model_tf(const DNNModel *model, DNNData *outputs, uint32_t nb_output);

void ff_dnn_free_model_tf(DNNModel **model);

typedef struct TFModel{
    TF_Graph *graph;
    TF_Session *session;
    TF_Status *status;
    TF_Output input;
    TF_Tensor *input_tensor;
    TF_Output *outputs;
    TF_Tensor **output_tensors;
    uint32_t nb_output;
} TFModel;

static void free_buffer(void *data, size_t length)
{
    av_freep(&data);
}

static TF_Buffer *read_graph(const char *model_filename)
{
    TF_Buffer *graph_buf;
    unsigned char *graph_data = NULL;
    AVIOContext *model_file_context;
    long size, bytes_read;

    if (avio_open(&model_file_context, model_filename, AVIO_FLAG_READ) < 0){
        return NULL;
    }

    size = avio_size(model_file_context);

    graph_data = av_malloc(size);
    if (!graph_data){
        avio_closep(&model_file_context);
        return NULL;
    }
    bytes_read = avio_read(model_file_context, graph_data, size);
    avio_closep(&model_file_context);
    if (bytes_read != size){
        av_freep(&graph_data);
        return NULL;
    }

    graph_buf = TF_NewBuffer();
    graph_buf->data = (void *)graph_data;
    graph_buf->length = size;
    graph_buf->data_deallocator = free_buffer;

    return graph_buf;
}

static TF_Tensor *allocate_input_tensor(const DNNData *input)
{
    TF_DataType dt;
    size_t size;
    int64_t input_dims[] = {1, input->height, input->width, input->channels};
    switch (input->dt) {
    case DNN_FLOAT:
        dt = TF_FLOAT;
        size = sizeof(float);
        break;
    case DNN_UINT8:
        dt = TF_UINT8;
        size = sizeof(char);
        break;
    default:
        //av_assert0(!"should not reach here");
        break;
    }

    return TF_AllocateTensor(dt, input_dims, 4,
                             input_dims[1] * input_dims[2] * input_dims[3] * size);
}

static DNNReturnType get_input_tf(void *model, DNNData *input, const char *input_name)
{
    TFModel *tf_model = (TFModel *)model;
    TF_Status *status;
    int64_t dims[4];

    TF_Output tf_output;
    tf_output.oper = TF_GraphOperationByName(tf_model->graph, input_name);
    if (!tf_output.oper)
        return DNN_ERROR;

    tf_output.index = 0;
    input->dt = TF_OperationOutputType(tf_output);

    status = TF_NewStatus();
    TF_GraphGetTensorShape(tf_model->graph, tf_output, dims, 4, status);
    if (TF_GetCode(status) != TF_OK){
        TF_DeleteStatus(status);
        return DNN_ERROR;
    }
    TF_DeleteStatus(status);

    //currently only NHWC is supported
    //some case is = -1
    //av_assert0(dims[0] == 1);    
    input->height = dims[1];
    input->width = dims[2];
    input->channels = dims[3];

    return DNN_SUCCESS;
}

static DNNReturnType set_input_output_tf(void *model, DNNData *input, const char *input_name, const char **output_names, uint32_t nb_output,uint32_t gpuid)
{
    TFModel *tf_model = (TFModel *)model;
    TF_SessionOptions *sess_opts;
    const TF_Operation *init_op = TF_GraphOperationByName(tf_model->graph, "init");

    // Input operation
    tf_model->input.oper = TF_GraphOperationByName(tf_model->graph, input_name);
    if (!tf_model->input.oper){
        return DNN_ERROR;
    }
    tf_model->input.index = 0;
    if (tf_model->input_tensor){
        TF_DeleteTensor(tf_model->input_tensor);
    }
    tf_model->input_tensor = allocate_input_tensor(input);
    if (!tf_model->input_tensor){
        return DNN_ERROR;
    }
    input->data = (float *)TF_TensorData(tf_model->input_tensor);

    // Output operation
    if (nb_output == 0)
        return DNN_ERROR;

    av_freep(&tf_model->outputs);
    tf_model->outputs = av_malloc_array(nb_output, sizeof(TF_Output));
    if (!tf_model->outputs)
        return DNN_ERROR;
    for (int i = 0; i < nb_output; ++i) {
        tf_model->outputs[i].oper = TF_GraphOperationByName(tf_model->graph, output_names[i]);
        if (!tf_model->outputs[i].oper){
            av_freep(&tf_model->outputs);
            return DNN_ERROR;
        }
        tf_model->outputs[i].index = 0;
    }

    if (tf_model->output_tensors) {
        for (uint32_t i = 0; i < tf_model->nb_output; ++i) {
            if (tf_model->output_tensors[i]) {
                TF_DeleteTensor(tf_model->output_tensors[i]);
                tf_model->output_tensors[i] = NULL;
            }
        }
    }
    av_freep(&tf_model->output_tensors);
    tf_model->output_tensors = av_mallocz_array(nb_output, sizeof(*tf_model->output_tensors));
    if (!tf_model->output_tensors) {
        av_freep(&tf_model->outputs);
        return DNN_ERROR;
    }

    tf_model->nb_output = nb_output;

    if (tf_model->session){
        TF_CloseSession(tf_model->session, tf_model->status);
        TF_DeleteSession(tf_model->session, tf_model->status);
    }
    sess_opts = TF_NewSessionOptions();
    // protobuf data for auto memory gpu_options.allow_growth=True and gpu_options.visible_device_list="0" 
    uint8_t config[10] = { 0x32, 0x5, 0x20, 0x1, 0x2a, 0x01, 0x30, 0x00, 0x00, }; 
    //config[6] += gpuid;  
    TF_SetConfig(sess_opts, (void*)config, 7, tf_model->status);

    tf_model->session = TF_NewSession(tf_model->graph, sess_opts, tf_model->status);
    TF_DeleteSessionOptions(sess_opts);
    if (TF_GetCode(tf_model->status) != TF_OK)
    {
        return DNN_ERROR;
    }

    // Run initialization operation with name "init" if it is present in graph
    if (init_op){
        TF_SessionRun(tf_model->session, NULL,
                      NULL, NULL, 0,
                      NULL, NULL, 0,
                      &init_op, 1, NULL, tf_model->status);
        if (TF_GetCode(tf_model->status) != TF_OK)
        {
            //av_log(NULL, AV_LOG_ERROR, "%d %d\n", __LINE__, TF_GetCode(tf_model->status));
            return DNN_ERROR;            
        }
    }

    return DNN_SUCCESS;
}

static DNNReturnType load_tf_model(TFModel *tf_model, const char *model_filename)
{
    TF_Buffer *graph_def;
    TF_ImportGraphDefOptions *graph_opts;

    graph_def = read_graph(model_filename);
    if (!graph_def){
        return DNN_ERROR;
    }
    tf_model->graph = TF_NewGraph();
    tf_model->status = TF_NewStatus();
    graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(tf_model->graph, graph_def, graph_opts, tf_model->status);
    TF_DeleteImportGraphDefOptions(graph_opts);
    TF_DeleteBuffer(graph_def);
    if (TF_GetCode(tf_model->status) != TF_OK){
        TF_DeleteGraph(tf_model->graph);
        TF_DeleteStatus(tf_model->status);
        return DNN_ERROR;
    }

    return DNN_SUCCESS;
}

DNNModel *ff_dnn_load_model_tf(const char *model_filename)
{
    DNNModel *model = NULL;
    TFModel *tf_model = NULL;

    model = av_malloc(sizeof(DNNModel));
    if (!model){
        return NULL;
    }

    tf_model = av_mallocz(sizeof(TFModel));
    if (!tf_model){
        av_freep(&model);
        return NULL;
    }

    if (load_tf_model(tf_model, model_filename) != DNN_SUCCESS){
        return NULL;        
    }

    model->model = (void *)tf_model;
    model->set_input_output = &set_input_output_tf;
    model->get_input = &get_input_tf;

    return model;
}

DNNReturnType ff_dnn_execute_model_tf(const DNNModel *model, DNNData *outputs, uint32_t nb_output)
{
    TFModel *tf_model = (TFModel *)model->model;
    uint32_t nb = FFMIN(nb_output, tf_model->nb_output);
    if (nb == 0)
        return DNN_ERROR;

    //av_assert0(tf_model->output_tensors);
    for (uint32_t i = 0; i < tf_model->nb_output; ++i) {
        if (tf_model->output_tensors[i]) {
            TF_DeleteTensor(tf_model->output_tensors[i]);
            tf_model->output_tensors[i] = NULL;
        }
    }

    TF_SessionRun(tf_model->session, NULL,
                  &tf_model->input, &tf_model->input_tensor, 1,
                  tf_model->outputs, tf_model->output_tensors, nb,
                  NULL, 0, NULL, tf_model->status);

    if (TF_GetCode(tf_model->status) != TF_OK){
        return DNN_ERROR;
    }

    for (uint32_t i = 0; i < nb; ++i) {
        outputs[i].height = TF_Dim(tf_model->output_tensors[i], 1);
        outputs[i].width = TF_Dim(tf_model->output_tensors[i], 2);
        outputs[i].channels = TF_Dim(tf_model->output_tensors[i], 3);
        outputs[i].data = TF_TensorData(tf_model->output_tensors[i]);
        outputs[i].dt = TF_TensorType(tf_model->output_tensors[i]);
    }

    return DNN_SUCCESS;
}

void ff_dnn_free_model_tf(DNNModel **model)
{
    TFModel *tf_model;

    if (*model){
        tf_model = (TFModel *)(*model)->model;
        if (tf_model->graph){
            TF_DeleteGraph(tf_model->graph);
        }
        if (tf_model->session){
            TF_CloseSession(tf_model->session, tf_model->status);
            TF_DeleteSession(tf_model->session, tf_model->status);
        }
        if (tf_model->status){
            TF_DeleteStatus(tf_model->status);
        }
        if (tf_model->input_tensor){
            TF_DeleteTensor(tf_model->input_tensor);
        }
        if (tf_model->output_tensors) {
            for (uint32_t i = 0; i < tf_model->nb_output; ++i) {
                if (tf_model->output_tensors[i]) {
                    TF_DeleteTensor(tf_model->output_tensors[i]);
                    tf_model->output_tensors[i] = NULL;
                }
            }
        }
        av_freep(&tf_model->outputs);
        av_freep(&tf_model->output_tensors);
        av_freep(&tf_model);
        av_freep(model);
    }
}

static DNNModule *get_dnn_module(DNNBackendType backend_type)
{
    DNNModule *dnn_module;

    dnn_module = av_malloc(sizeof(DNNModule));
    if(!dnn_module){
        return NULL;
    }

    switch(backend_type){
    case DNN_TF:
        dnn_module->load_model = &ff_dnn_load_model_tf;
        dnn_module->execute_model = &ff_dnn_execute_model_tf;
        dnn_module->free_model = &ff_dnn_free_model_tf;
        break;
    default:
        av_log(NULL, AV_LOG_ERROR, "Module backend_type is not native or tensorflow\n");
        av_freep(&dnn_module);
        return NULL;
    }

    return dnn_module;
}

static int copy_from_frame_to_dnn(LVPDnnContext *ctx, const AVFrame *frame)
{
    if(ctx == NULL || ctx->swscaleframe == NULL || 
      ctx->sws_rgb_scale == NULL || ctx->sws_gray8_to_grayf32 == NULL) return DNN_ERROR;

    int bytewidth = av_image_get_linesize(ctx->swscaleframe->format, ctx->swscaleframe->width, 0);
    DNNData *dnn_input = &ctx->input;

    if(ctx->swframeforHW)
    {
        if(av_hwframe_transfer_data(ctx->swframeforHW, frame, 0) != 0)
            return AVERROR(EIO);
        
        sws_scale(ctx->sws_rgb_scale, (const uint8_t **)ctx->swframeforHW->data, ctx->swframeforHW->linesize,
                  0, ctx->swframeforHW->height, (uint8_t * const*)(&ctx->swscaleframe->data),
                 ctx->swscaleframe->linesize);

    }
    else
    {
        sws_scale(ctx->sws_rgb_scale, (const uint8_t **)frame->data, frame->linesize,
                  0, frame->height, (uint8_t * const*)(&ctx->swscaleframe->data),
                  ctx->swscaleframe->linesize);
    }


    if (dnn_input->dt == DNN_FLOAT) {
        
        sws_scale(ctx->sws_gray8_to_grayf32, (const uint8_t **)ctx->swscaleframe->data, ctx->swscaleframe->linesize,
                    0, ctx->swscaleframe->height, (uint8_t * const*)(&dnn_input->data),
                    (const int [4]){ctx->swscaleframe->width * 3 * sizeof(float), 0, 0, 0});

        if(ctx->filter_type == DNN_YOLO) 
        {
          for (int i = 0; i < ctx->swscaleframe->height; i++)
          {
            float* pfrgb = ((float*)dnn_input->data) + ctx->swscaleframe->width * 3 * i;
            //uint8_t* pirgb = ctx->swscaleframe->data[i];

            for (int j = 0; j < ctx->swscaleframe->width * 3; j++)
            {
              *pfrgb = (float)(*pfrgb) * 255.0; pfrgb++;
               //pirgb++;
            }            
          }
        }

    } else {
        //av_assert0(dnn_input->dt == DNN_UINT8);
        av_image_copy_plane(dnn_input->data, bytewidth,
                            ctx->swscaleframe->data[0], ctx->swscaleframe->linesize[0],
                            bytewidth, ctx->swscaleframe->height);
    }
    
    return 0;   
}

LVPDnnContext* pgdnncontext = NULL;
static enum AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;

static int  lpms_detectoneframe(LVPDnnContext *ctx, AVFrame *in,int flagclass, float *fconfidence)
{ 

  char slvpinfo[256] = {0,};
  *fconfidence = 0.0;
  int dnn_result;
  if(ctx == NULL) return DNN_ERROR;
  //ctx->framenum = 1;
  //if(ctx->sample_rate > 0 && ctx->framenum % ctx->sample_rate == 0 &&
  //  copy_from_frame_to_dnn(ctx, in) == DNN_SUCCESS)

  if(copy_from_frame_to_dnn(ctx, in) != DNN_SUCCESS) return DNN_ERROR;

  dnn_result = (ctx->dnn_module->execute_model)(ctx->model, &ctx->output, 1);
  if (dnn_result != DNN_SUCCESS){
      av_log(NULL, AV_LOG_ERROR, "failed to execute model\n");
      return AVERROR(EIO);
  }

  DNNData *dnn_output = &ctx->output;
  float* pfdata = dnn_output->data;
  int lendata = ctx->output.height;
  //            
  if(flagclass >= 0 && flagclass < lendata /*&& pfdata[0] >= ctx->valid_threshold*/)
  {
      *fconfidence = pfdata[flagclass];
      //snprintf(slvpinfo, sizeof(slvpinfo), "probability %.2f", pfdata[0]);  
      
      //av_dict_set(metadata, "lavfi.lvpdnn.text", slvpinfo, 0);
      if(ctx->logfile)
      {
          fprintf(ctx->logfile,"%s\n",slvpinfo);                
      }      
  }
  else {
    av_log(0, AV_LOG_INFO, "invalid classification numbel %d\n",flagclass);
  }
  
  //for DEBUG
  //av_log(0, AV_LOG_INFO, "%d frame detected as %s confidence\n",ctx->framenum,slvpinfo);

  if(ctx->logfile && ctx->framenum % 20 == 0)
      fflush(ctx->logfile);
      
  return dnn_result;
}
static int  lpms_detectoneframewithctx(LVPDnnContext *ctx, AVFrame *in)
{   
  int dnn_result;
  if(ctx == NULL) return DNN_ERROR;
  //ctx->framenum = 1;
  //if(ctx->sample_rate > 0 && ctx->framenum % ctx->sample_rate == 0 &&
  //  copy_from_frame_to_dnn(ctx, in) == DNN_SUCCESS)

  if(copy_from_frame_to_dnn(ctx, in) != DNN_SUCCESS) {
    av_log(NULL, AV_LOG_ERROR, "frame_to_dnn function failed for classification\n");
    return DNN_ERROR;
  }

  dnn_result = (ctx->dnn_module->execute_model)(ctx->model, &ctx->output, 1);
  if (dnn_result != DNN_SUCCESS){
      av_log(NULL, AV_LOG_ERROR, "failed to execute model\n");
      return AVERROR(EIO);
  }
 switch (ctx->filter_type)
  {
  case DNN_CLASSIFY:
    if (ctx->fmatching != NULL) {
      float* pfdata = (float*)ctx->output.data;
      for (int k = 0; k < ctx->output.height; k++)
      {
        ctx->fmatching[k] += pfdata[k];
      }
      ctx->runcount ++;
      //for DEBUG
      //av_log(0, AV_LOG_INFO, "classification num runcount = %d %d\n",ctx->output.height,ctx->runcount);
    }
    break;
  case DNN_YOLO:    
    if(ctx->boxes != NULL && ctx->probs != NULL ){
      //init buffer
       memset(ctx->boxes, 0x00, ctx->output.height*sizeof(box));	    
	    for (int j = 0; j < ctx->output.height; ++j) 
        memset(ctx->probs[j], 0x00, ctx->classes * sizeof(float));
      memset(ctx->object, 0x00, ctx->output.height*sizeof(boxobject));

      //av_log(0, AV_LOG_ERROR, "yolo detect = %d %d\n",ctx->output.height,ctx->classes);
      int objecount = 0;
      float xscale = in->width / (float)ctx->input.width;
      float yscale = in->height / (float)ctx->input.height;
      layer ldata = { 1, ctx->output.height, ctx->output.width, 0, ctx->classes };
      float* pfdata = (float*)ctx->output.data;

      if (get_detection_boxes(pfdata, ldata, 1, 1, 0.7, ctx->probs, ctx->boxes, 0) > 0){
        do_nms(ctx->boxes, ctx->probs, ldata.n, ldata.classes, 0.4);	
        get_detections(ctx, in->width, in->height, xscale, yscale, ldata.n, 0.5, ctx->boxes, 
        ctx->probs, ldata.classes, ctx->object, &objecount);
      }
      ctx->runcount ++;      
    }
    break;
  default:
    break;
  }

  /*
  char slvpinfo[256] = {0,};
  float* pfdata = ctx->output.data;
  int lendata = ctx->output.height;

  //get confidence order  
  for (int i = 0; i < lendata; i++)
  {
      if(pfdata[i] > *fconfidence) {
        *fconfidence = pfdata[i];
        *classid = i;
      }
  }  
  //for DEBUG
  //av_log(0, AV_LOG_INFO, "classification id confidence = %d %f\n",*classid,*fconfidence);
  //need some code for metadata
  //if(ctx->logfile && ctx->framenum % 20 == 0)
  //    fflush(ctx->logfile);
  */  
  return dnn_result;
}

static int hw_decoder_init(LVPDnnContext* lvpctx, const enum AVHWDeviceType type)
{
    int err = 0;
	AVCodecContext *ctx = lvpctx->decoder_ctx;
	if(ctx == NULL) return -1;

    if ((err = av_hwdevice_ctx_create(&lvpctx->hw_device_ctx, type,NULL, NULL, 0)) < 0){
        fprintf(stderr, "Failed to create specified HW device.\n");
        return err;
    }
    ctx->hw_device_ctx = av_buffer_ref(lvpctx->hw_device_ctx);

    return err;
}
static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat *pix_fmts)
{
    const enum AVPixelFormat *p;
	//AVCodecContext *ctx = lvpctx->decoder_ctx; 
	//if(ctx == NULL || pgdnncontext == NULL) return AV_PIX_FMT_NONE;
  if(ctx == NULL || hw_pix_fmt == AV_PIX_FMT_NONE) return AV_PIX_FMT_NONE;

    for (p = pix_fmts; *p != -1; p++) {
        if (*p == hw_pix_fmt)
            return *p;
    }

    fprintf(stderr, "Failed to get HW surface format.\n");
    return AV_PIX_FMT_NONE;
}

static int dnn_decodeframe(AVCodecContext *avctx,AVPacket *pkt, AVFrame *frame, int *got_frame)
{
    int ret;

    *got_frame = 0;
    if (pkt) {
        ret = avcodec_send_packet(avctx, pkt);
        if (ret < 0 && ret != AVERROR_EOF)
            return ret;
    }
    ret = avcodec_receive_frame(avctx, frame);
    if (ret < 0 && ret != AVERROR(EAGAIN))
        return ret;
    if (ret >= 0)
        *got_frame = 1;
    return 0;
}

static int prepare_sws_context(LVPDnnContext *ctx, AVFrame *frame, int flagHW)
{
	int ret = 0;

	enum AVPixelFormat fmt = 0;
	
	if(flagHW){	
  	//fmt = ctx->hw_pix_fmt;
    enum AVPixelFormat *formats;

    ret = av_hwframe_transfer_get_formats(frame->hw_frames_ctx,
                                        AV_HWFRAME_TRANSFER_DIRECTION_FROM,
                                        &formats, 0);
    if (ret < 0) {        
        return ret;
    }
			
	  fmt = formats[0];                
    av_freep(&formats);

  }
	else 
		fmt = frame->format;
	
    ctx->sws_rgb_scale = sws_getContext(frame->width, frame->height, fmt,
                                            ctx->input.width, ctx->input.height, AV_PIX_FMT_RGB24,
                                            SWS_BILINEAR, NULL, NULL, NULL);

    ctx->sws_gray8_to_grayf32 = sws_getContext(ctx->input.width*3,
                                                ctx->input.height,
                                                AV_PIX_FMT_GRAY8,
                                                ctx->input.width*3,
                                                ctx->input.height,
                                                AV_PIX_FMT_GRAYF32,
                                                0, NULL, NULL, NULL);  

  

    if(ctx->sws_rgb_scale == 0 || ctx->sws_gray8_to_grayf32 == 0)
    {
        av_log(0, AV_LOG_ERROR, "could not create scale context\n");
        return AVERROR(ENOMEM);
    }

    ctx->swscaleframe = av_frame_alloc();
    if (!ctx->swscaleframe)
        return AVERROR(ENOMEM);

    ctx->swscaleframe->format = AV_PIX_FMT_RGB24;
    ctx->swscaleframe->width  = ctx->input.width;
    ctx->swscaleframe->height = ctx->input.height;
    ret = av_frame_get_buffer(ctx->swscaleframe, 0);
    if (ret < 0) {
        av_frame_free(&ctx->swscaleframe);
        return ret;
    }
    if (flagHW){
        ctx->swframeforHW = av_frame_alloc();
        if (!ctx->swframeforHW)
        return AVERROR(ENOMEM);
    }
	
	return ret;
}

int  lpms_dnnexecute(char* ivpath, int  flagHW, int  flagclass, float  tinteval,float* porob)
{
	char sdevicetype[64] = {0,};
	int	 ret, i;
	AVStream *video = NULL;
	AVPacket packet;

	LVPDnnContext *context = pgdnncontext;
	if(context == NULL) return DNN_ERROR;	

	*porob = 0.0;

	if(flagHW){
		strcpy(sdevicetype,"cuda");
		context->type = av_hwdevice_find_type_by_name(sdevicetype);
		if (context->type == AV_HWDEVICE_TYPE_NONE) return DNN_ERROR;
	}

	/* open the input file */
  if (avformat_open_input(&context->input_ctx, ivpath, NULL, NULL) != 0) {
      fprintf(stderr, "Cannot open input file '%s'\n", ivpath);
      return DNN_ERROR;
  }

  if (avformat_find_stream_info(context->input_ctx, NULL) < 0) {
      fprintf(stderr, "Cannot find input stream information.\n");
      return DNN_ERROR;
  }

  /* find the video stream information */
  ret = av_find_best_stream(context->input_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &context->decoder, 0);
  if (ret < 0) {
      fprintf(stderr, "Cannot find a video stream in the input file\n");
      return DNN_ERROR;
  }
  context->video_stream = ret;
	
	if(flagHW){
		for (i = 0;; i++) {
	        const AVCodecHWConfig *config = avcodec_get_hw_config(context->decoder, i);
	        if (!config) {
	            fprintf(stderr, "Decoder %s does not support device type %s.\n",
	                    context->decoder->name, av_hwdevice_get_type_name(context->type));
	            return -1;
	        }
	        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
	            config->device_type == context->type) {
	            //context->hw_pix_fmt = config->pix_fmt;
              hw_pix_fmt = config->pix_fmt;
	            break;
	        }
    	}
	}	
	if (!(context->decoder_ctx = avcodec_alloc_context3(context->decoder)))
			return AVERROR(ENOMEM);
	
	video = context->input_ctx->streams[context->video_stream];
	if (avcodec_parameters_to_context(context->decoder_ctx, video->codecpar) < 0)
		return -1;

	if(flagHW){
		context->decoder_ctx->get_format  = get_hw_format;
		if (hw_decoder_init(context, context->type) < 0)
			return -1;
	}

	if ((ret = avcodec_open2(context->decoder_ctx, context->decoder, NULL)) < 0) {
        fprintf(stderr, "Failed to open codec for stream #%u\n", context->video_stream);
        return -1;
    }

	/* actual decoding and dump the raw data */
	context->framenum = 0;
	int ngotframe = 0;
  float fconfidence ,ftotal = 0.0;
  int dnnresult, count = 0;

  /*determine sample rate according to input video's frame rate*/
  float frmarate = 1.0;
  if(video->r_frame_rate.den > 0.0){
    frmarate = av_q2d(video->r_frame_rate);
  } else  {
    frmarate = 1.0 / av_q2d(video->time_base);    
  }  
  int nsamplerate = (int)(frmarate * tinteval);
  if(nsamplerate == 0) nsamplerate = context->sample_rate;

	context->readframe = av_frame_alloc();

  while (ret >= 0) {
		if ((ret = av_read_frame(context->input_ctx, &packet)) < 0)
	    	break;

	    if (context->video_stream == packet.stream_index)
	    {	
	      ret = dnn_decodeframe(context->decoder_ctx,&packet,context->readframe,&ngotframe);
        if(ret < 0 || ngotframe == 0)
          continue;

        if(context->sws_rgb_scale == NULL || context->sws_gray8_to_grayf32 == NULL)
        {
          ret = prepare_sws_context(context,context->readframe,flagHW);
          if(ret < 0){
            av_log(NULL, AV_LOG_INFO, "Can not create scale context!\n");
            break;
          }
        }
        context->framenum ++;
        if(context->framenum % nsamplerate == 0){
          dnnresult = lpms_detectoneframe(context,context->readframe,flagclass,&fconfidence);
          if(dnnresult == DNN_SUCCESS){
            count++;
            ftotal += fconfidence;
          }
        }
			  av_frame_unref(context->readframe);
	    }
		
	    av_packet_unref(&packet);
	}

  if(count)
  	*porob = ftotal / count;

  av_log(0, AV_LOG_INFO, "Engine Probability = %f\n",*porob);
  
  //release frame and scale context
  if(context->readframe)
		  av_frame_free(&context->readframe);
  if(context->swscaleframe)
      av_frame_free(&context->swscaleframe);
  if(context->swframeforHW)
      av_frame_free(&context->swframeforHW);

  sws_freeContext(context->sws_rgb_scale);
  context->sws_rgb_scale = NULL;
  sws_freeContext(context->sws_gray8_to_grayf32);
  context->sws_gray8_to_grayf32 = NULL;
  //release avcontext

	avcodec_free_context(&context->decoder_ctx);
	avformat_close_input(&context->input_ctx);
	av_buffer_unref(&context->hw_device_ctx);	

	return DNN_SUCCESS;
}

int  lpms_dnninit(char* fmodelpath, char* input, char* output, int samplerate, float fthreshold)
{   
  DNNReturnType result;
  DNNData model_input;
  int check;
  if(fmodelpath == NULL) return DNN_ERROR;
  if(pgdnncontext != NULL) return DNN_SUCCESS;
  
  LVPDnnContext *ctx = (LVPDnnContext*)av_mallocz(sizeof(LVPDnnContext));
  pgdnncontext = ctx;

  ctx->model_filename = (char*)malloc(MAXPATH);
  ctx->model_inputname = (char*)malloc(MAXPATH);
  ctx->model_outputname = (char*)malloc(MAXPATH);
  strcpy(ctx->model_filename,fmodelpath);
	strcpy(ctx->model_inputname,input);
	strcpy(ctx->model_outputname,output);
  ctx->sample_rate = samplerate;
  ctx->valid_threshold = fthreshold;


  if (strlen(ctx->model_filename)<=0) {
      av_log(NULL, AV_LOG_ERROR, "model file for network is not specified\n");
      return AVERROR(EINVAL);
  }
  if (strlen(ctx->model_inputname)<=0) {
      av_log(NULL, AV_LOG_ERROR, "input name of the model network is not specified\n");
      return AVERROR(EINVAL);
  }
  if (strlen(ctx->model_outputname)<=0) {
      av_log(NULL, AV_LOG_ERROR, "output name of the model network is not specified\n");
      return AVERROR(EINVAL);
  }

  if (strlen(ctx->log_filename)<=0) {
      av_log(NULL, AV_LOG_INFO, "output file for log is not specified\n");
      //return AVERROR(EINVAL);
  }

  ctx->backend_type = 1;
  ctx->dnn_module = get_dnn_module(ctx->backend_type);
  if (!ctx->dnn_module) {
      av_log(NULL, AV_LOG_ERROR, "could not create DNN module for requested backend\n");
      return AVERROR(ENOMEM);
  }
  if (!ctx->dnn_module->load_model) {
      av_log(NULL, AV_LOG_ERROR, "load_model for network is not specified\n");
      return AVERROR(EINVAL);
  }

  ctx->model = (ctx->dnn_module->load_model)(ctx->model_filename);
  if (!ctx->model) {
      av_log(NULL, AV_LOG_ERROR, "could not load DNN model\n");
      return AVERROR(EINVAL);
  }

  if(strlen(ctx->log_filename) > 0){        
      ctx->logfile = fopen(ctx->log_filename, "w");
  }
  else{        
      ctx->logfile = NULL;
  }

  ctx->framenum = 0;
  //config input

  result = ctx->model->get_input(ctx->model->model, &model_input, ctx->model_inputname);
  if (result != DNN_SUCCESS) {
      av_log(NULL, AV_LOG_ERROR, "could not get input from the model\n");
      return AVERROR(EIO);
  }

  ctx->input.width    = model_input.width;
  ctx->input.height   = model_input.height;
  ctx->input.channels = model_input.channels;
  ctx->input.dt = model_input.dt;

  result = (ctx->model->set_input_output)(ctx->model->model,
                                      &ctx->input, ctx->model_inputname,
                                      (const char **)&ctx->model_outputname, 1 , 0);
  

  if (result != DNN_SUCCESS) {
      av_log(NULL, AV_LOG_ERROR, "could not set input and output for the model\n");
      return AVERROR(EIO);
  }

  // have a try run in case that the dnn model resize the frame
  result = (ctx->dnn_module->execute_model)(ctx->model, &ctx->output, 1);
  if (result != DNN_SUCCESS){
      av_log(NULL, AV_LOG_ERROR, "failed to execute model\n");
      return AVERROR(EIO);
  }
  
  return DNN_SUCCESS;
}

void  lpms_dnnfree()
{
  LVPDnnContext *context = pgdnncontext;

  if(context == NULL) return;

  if(context->sws_rgb_scale)
  sws_freeContext(context->sws_rgb_scale);
  if(context->sws_gray8_to_grayf32)
  sws_freeContext(context->sws_gray8_to_grayf32);  

  if (context->dnn_module)
      (context->dnn_module->free_model)(&context->model);

  av_freep(&context->dnn_module);

  if(context->readframe)
		av_frame_free(&context->readframe);

  if(context->swscaleframe)
      av_frame_free(&context->swscaleframe);

  if(context->swframeforHW)
      av_frame_free(&context->swframeforHW);

  if(strlen(context->log_filename) > 0 && context->logfile)
  {
      fclose(context->logfile);
  }

  if(context->model_filename)
    free(context->model_filename);
  if(context->model_inputname)
    free(context->model_inputname);
  if(context->model_outputname)
    free(context->model_outputname);
  if(pgdnncontext)
    av_free(pgdnncontext);
  pgdnncontext = NULL;
    
}

//for multiple model 
int  lpms_dnninitwithctx(LVPDnnContext* ctx, char* fmodelpath, char* input, char* output, int samplerate, float fthreshold, int gpuid)
{   
  DNNReturnType result;
  DNNData model_input;
  int check;
  if(ctx ==NULL || fmodelpath == NULL) return DNN_ERROR;  
  
  ctx->model_filename = (char*)malloc(MAXPATH);
  ctx->model_inputname = (char*)malloc(MAXPATH);
  ctx->model_outputname = (char*)malloc(MAXPATH);
  strcpy(ctx->model_filename,fmodelpath);
	strcpy(ctx->model_inputname,input);
	strcpy(ctx->model_outputname,output);
  ctx->sample_rate = samplerate;
  ctx->valid_threshold = fthreshold;

  if(gpuid >= 0) ctx->gpuid = gpuid;
  else ctx->gpuid = 0;

  if (strlen(ctx->model_filename)<=0) {
      av_log(NULL, AV_LOG_ERROR, "model file for network is not specified\n");
      return AVERROR(EINVAL);
  }
  if (strlen(ctx->model_inputname)<=0) {
      av_log(NULL, AV_LOG_ERROR, "input name of the model network is not specified\n");
      return AVERROR(EINVAL);
  }
  if (strlen(ctx->model_outputname)<=0) {
      av_log(NULL, AV_LOG_ERROR, "output name of the model network is not specified\n");
      return AVERROR(EINVAL);
  }

  if (strlen(ctx->log_filename)<=0) {
      av_log(NULL, AV_LOG_INFO, "output file for log is not specified\n");
      //return AVERROR(EINVAL);
  }

  ctx->backend_type = 1;
  ctx->dnn_module = get_dnn_module(ctx->backend_type);
  if (!ctx->dnn_module) {
      av_log(NULL, AV_LOG_ERROR, "could not create DNN module for requested backend\n");
      return AVERROR(ENOMEM);
  }
  if (!ctx->dnn_module->load_model) {
      av_log(NULL, AV_LOG_ERROR, "load_model for network is not specified\n");
      return AVERROR(EINVAL);
  }

  ctx->model = (ctx->dnn_module->load_model)(ctx->model_filename);
  if (!ctx->model) {
      av_log(NULL, AV_LOG_ERROR, "could not load DNN model\n");
      return AVERROR(EINVAL);
  }

  if(strlen(ctx->log_filename) > 0){        
      ctx->logfile = fopen(ctx->log_filename, "w");
  }
  else{        
      ctx->logfile = NULL;
  }

  ctx->framenum = 0;
  //config input

  result = ctx->model->get_input(ctx->model->model, &model_input, ctx->model_inputname);
  if (result != DNN_SUCCESS) {
      av_log(NULL, AV_LOG_ERROR, "could not get input from the model\n");
      return AVERROR(EIO);
  }

  ctx->input.width    = model_input.width;
  ctx->input.height   = model_input.height;
  ctx->input.channels = model_input.channels;
  ctx->input.dt = model_input.dt;

  result = (ctx->model->set_input_output)(ctx->model->model,
                                      &ctx->input, ctx->model_inputname,
                                      (const char **)&ctx->model_outputname, 1,ctx->gpuid);
  

  if (result != DNN_SUCCESS) {
      av_log(NULL, AV_LOG_ERROR, "could not set input and output for the model\n");
      return AVERROR(EIO);
  }

  // have a try run in case that the dnn model resize the frame
  result = (ctx->dnn_module->execute_model)(ctx->model, &ctx->output, 1);
  if (result != DNN_SUCCESS){
      av_log(NULL, AV_LOG_ERROR, "failed to execute model\n");
      return AVERROR(EIO);
  }

  switch (ctx->filter_type)
  {
  case DNN_CLASSIFY:
    ctx->fmatching = (float*)malloc((ctx->output.height + 1) * sizeof(float));
    if(ctx->fmatching == NULL){
        av_log(NULL, AV_LOG_ERROR, "can not creat matching buffer\n");
        return AVERROR(EIO);
    }
    memset(ctx->fmatching, 0x00, (ctx->output.height + 1) * sizeof(float));
    break;

  case DNN_YOLO:
      av_log(NULL, AV_LOG_ERROR, "yolo information %d %d %d\n",
      ctx->output.width,ctx->output.height,ctx->output.channels);      
      ctx->classes = ctx->output.width - 5;
      ctx->boxes = (box*)calloc(ctx->output.height, sizeof(box));
	    ctx->probs = (float**)calloc(ctx->output.height, sizeof(float*));
	    for (int j = 0; j < ctx->output.height; ++j) ctx->probs[j] = (float*)calloc(ctx->classes, sizeof(float));
	    ctx->object = (boxobject*)calloc(ctx->output.height, sizeof(boxobject));
      //for result
      ctx->result = (char**)calloc(MAX_YOLO_FRAME, sizeof(char*));
	    for (int j = 0; j < MAX_YOLO_FRAME; ++j) ctx->result[j] = (char*)calloc(YOLO_FRESULTMAXPATH, sizeof(char));
      
    break;

  default:
    break;
  }
  
  //av_log(NULL, AV_LOG_ERROR, "lpms_dnninitwithctx model success\n");
  return DNN_SUCCESS;
}

void  lpms_dnnfreewithctx(LVPDnnContext *context)
{
  if(context == NULL) return;

  if(context->sws_rgb_scale)
  sws_freeContext(context->sws_rgb_scale);
  if(context->sws_gray8_to_grayf32)
  sws_freeContext(context->sws_gray8_to_grayf32);  

  if (context->dnn_module)
      (context->dnn_module->free_model)(&context->model);

  if(context->dnn_module)
    av_freep(&context->dnn_module);

  if(context->readframe)
		av_frame_free(&context->readframe);

  if(context->swscaleframe)
    av_frame_free(&context->swscaleframe);

  if(context->swframeforHW)
    av_frame_free(&context->swframeforHW);

  if(strlen(context->log_filename) > 0 && context->logfile)
  {
      fclose(context->logfile);
  }

  if(context->model_filename){
    free(context->model_filename);
    context->model_filename = NULL;
  }
  if(context->model_inputname){
    free(context->model_inputname);
    context->model_inputname = NULL;
  }    
  if(context->model_outputname){
    free(context->model_outputname);
    context->model_outputname = NULL;
  }
  switch (context->filter_type)
  {
  case DNN_CLASSIFY:
    if(context->fmatching){
      free(context->fmatching);
      context->fmatching = NULL;
    }
    break;
  case DNN_YOLO:      
    if(context->boxes){
      free(context->boxes);
      context->boxes = NULL;
    }
    if(context->object){
      free(context->object);
      context->object = NULL;
    }
    for (int j = 0; j < context->output.height; ++j) {
      if(context->probs[j]) free(context->probs[j]);        
    }
    if(context->probs){
      free(context->probs);
      context->probs = NULL;
    }
    //for result
    for (int j = 0; j < MAX_YOLO_FRAME; ++j) {
      if(context->result[j]) free(context->result[j]);        
    }
    if(context->result){
      free(context->result);
      context->result = NULL;
    }
    break;

  default:
    break;
  }

  if(context)
    av_free(context);

  context = NULL;
}
int  lpms_dnnexecutewithctx(LVPDnnContext *context, char* ivpath, int flagHW, float tinteval, int* classid, float* porob)
{
	char sdevicetype[64] = {0,};
	int	 ret, i , classnum;
	AVStream *video = NULL;
	AVPacket packet;
  float *confidences = NULL;
	
	if(context == NULL || ivpath == NULL) return DNN_ERROR;	

	*porob = 0.0;
  *classid = -1;

	if(flagHW){
		strcpy(sdevicetype,"cuda");
		context->type = av_hwdevice_find_type_by_name(sdevicetype);
		if (context->type == AV_HWDEVICE_TYPE_NONE) return DNN_ERROR;
	}

	/* open the input file */
    if (avformat_open_input(&context->input_ctx, ivpath, NULL, NULL) != 0) {
        fprintf(stderr, "Cannot open input file '%s'\n", ivpath);
        return DNN_ERROR;
    }

    if (avformat_find_stream_info(context->input_ctx, NULL) < 0) {
        fprintf(stderr, "Cannot find input stream information.\n");
        return DNN_ERROR;
    }

    /* find the video stream information */
    ret = av_find_best_stream(context->input_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &context->decoder, 0);
    if (ret < 0) {
        fprintf(stderr, "Cannot find a video stream in the input file\n");
        return DNN_ERROR;
    }
    context->video_stream = ret;
	
	if(flagHW) {
		for (i = 0;; i++) {
        const AVCodecHWConfig *config = avcodec_get_hw_config(context->decoder, i);
        if (!config) {
            fprintf(stderr, "Decoder %s does not support device type %s.\n",
                    context->decoder->name, av_hwdevice_get_type_name(context->type));
            return -1;
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
            config->device_type == context->type) {
            //context->hw_pix_fmt = config->pix_fmt;
            hw_pix_fmt = config->pix_fmt;
            break;
      } 
    }
	}	
	if (!(context->decoder_ctx = avcodec_alloc_context3(context->decoder)))
			return AVERROR(ENOMEM);
	
	video = context->input_ctx->streams[context->video_stream];
	if (avcodec_parameters_to_context(context->decoder_ctx, video->codecpar) < 0)
		return -1;

	if(flagHW){
		context->decoder_ctx->get_format  = get_hw_format;
		if (hw_decoder_init(context, context->type) < 0)
			return -1;
	}

	if ((ret = avcodec_open2(context->decoder_ctx, context->decoder, NULL)) < 0) {
        fprintf(stderr, "Failed to open codec for stream #%u\n", context->video_stream);
        return -1;
    }

	/* actual decoding and dump the raw data */
	context->framenum = 0;
	int ngotframe = 0;  
  int dnnresult, count = 0;

  /*determine sample rate according to input video's frame rate*/
  float frmarate = 1.0;
  if(video->r_frame_rate.den > 0.0){
    frmarate = av_q2d(video->r_frame_rate);
  } else  {
    frmarate = 1.0 / av_q2d(video->time_base);    
  }  
  int nsamplerate = (int)(frmarate * tinteval);
  if(nsamplerate == 0) nsamplerate = context->sample_rate;

  classnum = context->output.height;
  //if really nsamplerate is zero then only transcoding
  if(nsamplerate == 0 || classnum == 0)
    goto cleancontext;

  memset(context->fmatching, 0x00, classnum * sizeof(float));
  context->runcount = 0;

	context->readframe = av_frame_alloc();
  
  while (ret >= 0) {

		if ((ret = av_read_frame(context->input_ctx, &packet)) < 0)
	    	break;

	    if (context->video_stream == packet.stream_index)
	    {	
	      ret = dnn_decodeframe(context->decoder_ctx,&packet,context->readframe,&ngotframe);
        if(ret < 0 || ngotframe == 0)
          continue;

        if(context->sws_rgb_scale == NULL || context->sws_gray8_to_grayf32 == NULL)
        {
          ret = prepare_sws_context(context,context->readframe,flagHW);
          if(ret < 0){
            av_log(NULL, AV_LOG_INFO, "Can not create scale context!\n");
            break;
          }
        }
        context->framenum ++;
        if(context->framenum % nsamplerate == 0){
          dnnresult = lpms_detectoneframewithctx(context,context->readframe);
          count++;        
        }
			  av_frame_unref(context->readframe);
	    }
		
	    av_packet_unref(&packet);
	}
  if(context->filter_type == DNN_CLASSIFY){
    confidences = (float*)context->fmatching;
    //find max prob classid  
    for (int i = 0; i < context->output.height; i++)
    {
        if(confidences[i] > *porob) {
          *porob = confidences[i];
          *classid = i;
        }
    }
    if(context->runcount > 1) {
      *porob = *porob / (float)context->runcount;
    }

    av_log(0, AV_LOG_ERROR, "Engine Classid & Probability = %d %f\n",*classid, *porob);
  }
  /*
  //get confidence order  
  for (int i = 0; i < classnum; i++)
  {
      if(confidences[i] > *porob) {
        *porob = confidences[i];
        *classid = i;
      }
  }  
  */ 
 cleancontext: 
  if(confidences)
    free(confidences);
  //release frame and scale context
  if(context->readframe)
		  av_frame_free(&context->readframe);
  if(context->swscaleframe)
      av_frame_free(&context->swscaleframe);
  if(context->swframeforHW)
      av_frame_free(&context->swframeforHW);

  sws_freeContext(context->sws_rgb_scale);
  context->sws_rgb_scale = NULL;
  sws_freeContext(context->sws_gray8_to_grayf32);
  context->sws_gray8_to_grayf32 = NULL;
  //release avcontext

	avcodec_free_context(&context->decoder_ctx);
	avformat_close_input(&context->input_ctx);
	av_buffer_unref(&context->hw_device_ctx);	

	return DNN_SUCCESS;
}
LVPDnnContext*  lpms_dnnnew()
{
  LVPDnnContext *ctx = (LVPDnnContext*)av_mallocz(sizeof(LVPDnnContext));
  if(ctx) ctx->filter_type = DNN_CLASSIFY;
  return ctx;
}
void lpms_dnnstop(LVPDnnContext* context)
{
  lpms_dnnfreewithctx(context);  
}

Vinfo*  lpms_vinfonew()
{
  Vinfo *vinfo = (Vinfo*)malloc(sizeof(Vinfo));
  memset(vinfo, 0x00, sizeof(Vinfo));
  return vinfo;
}
int lpms_getvideoinfo(char* ivpath, Vinfo* vinfo)
{
  if(ivpath == NULL || vinfo == NULL) return DNN_ERROR;
  AVFormatContext 	*ic = NULL;		
	AVCodec 			    *decoder = NULL;
  AVCodecContext 		*dx = NULL;
  AVStream          *video = NULL;
  int 				      ret,video_stream;

  if (avformat_open_input(&ic, ivpath, NULL, NULL) != 0) {
      fprintf(stderr, "Cannot open input file '%s'\n", ivpath);
      return DNN_ERROR;
  }

  if (avformat_find_stream_info(ic, NULL) < 0) {
      fprintf(stderr, "Cannot find input stream information.\n");
      return DNN_ERROR;
  }

  /* find the video stream information */
  ret = av_find_best_stream(ic, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
  if (ret < 0) {
      fprintf(stderr, "Cannot find a video stream in the input file\n");
      return DNN_ERROR;
  }
 
  video_stream = ret;
  video = ic->streams[video_stream];

  if (!(dx = avcodec_alloc_context3(decoder))) return AVERROR(ENOMEM);
  if (avcodec_parameters_to_context(dx, video->codecpar) < 0) return DNN_ERROR;  

	if ((ret = avcodec_open2(dx, decoder, NULL)) < 0) {
        fprintf(stderr, "Failed to open codec for stream #%u\n", video_stream);
        return -1;
  }  

  vinfo->fps = 1.0;
  if(video->r_frame_rate.den > 0.0) {
    vinfo->fps = av_q2d(video->r_frame_rate);
  } else {
    vinfo->fps = 1.0 / av_q2d(video->time_base);    
  }  

  vinfo->width = dx->width;
  vinfo->height = dx->height;

  vinfo->duration = (double)ic->duration / (double)AV_TIME_BASE;
	if (vinfo->duration <= 0.000001) {
		vinfo->duration = (double)video->duration * av_q2d(video->time_base);
	}

	vinfo->framecount = video->nb_frames;  
	if (vinfo->framecount == 0) {
		vinfo->framecount = (int)(vinfo->duration * vinfo->fps + 0.5);
	}
 
  avcodec_free_context(&dx);
	avformat_close_input(&ic); 
	return DNN_SUCCESS;
}

#endif
