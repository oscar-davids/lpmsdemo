#ifndef _LPMS_FFMPEG_H_
#define _LPMS_FFMPEG_H_

#include <libavutil/hwcontext.h>
#include <libavutil/rational.h>

// LPMS specific errors
extern const int lpms_ERR_INPUT_PIXFMT;
extern const int lpms_ERR_FILTERS;
extern const int lpms_ERR_OUTPUTS;

struct transcode_thread;


typedef struct {
    char *name;
    AVDictionary *opts;
} component_opts;

typedef struct {
  char *fname;
  char *vfilters;
  int w, h, bitrate;
  AVRational fps;

  component_opts muxer;
  component_opts audio;
  component_opts video;

} output_params;

typedef struct {
  char *fname;

  // Handle to a transcode thread.
  // If null, a new transcode thread is allocated.
  // The transcode thread is returned within `output_results`.
  // Must be freed with lpms_transcode_stop.
  struct transcode_thread *handle;

  // Optional hardware acceleration
  enum AVHWDeviceType hw_type;
  char *device;
  //for ffmpeg metadata. if not null, write ffmpeg metadata.
  char *metadata;
} input_params;

typedef struct {
    int frames;
    int64_t pixels;
} output_results;


void lpms_init();
int  lpms_rtmp2hls(char *listen, char *outf, char *ts_tmpl, char *seg_time, char *seg_start);
int  lpms_transcode(input_params *inp, output_params *params, output_results *results, int nb_outputs, output_results *decoded_results);
struct transcode_thread* lpms_transcode_new();
void lpms_transcode_stop(struct transcode_thread* handle);

#define _ADD_LPMS_DNN_
#ifdef _ADD_LPMS_DNN_

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

#ifndef MAXPATH
#define MAXPATH 256
#endif

typedef enum {DNN_FLOAT = 1, DNN_UINT8 = 4} DNNDataType;
typedef enum {DNN_NATIVE, DNN_TF} DNNBackendType;
typedef enum {DNN_SUCCESS, DNN_ERROR} DNNReturnType;

typedef struct Vinfo{
    float fps;
    int width, height, channels;
    int framecount;
    float duration;
} Vinfo;

typedef struct DNNData{
    void *data;
    DNNDataType dt;
    int width, height, channels;
} DNNData;

typedef struct DNNModel{
    // Stores model that can be different for different backends.
    void *model;
    // Gets model input information
    // Just reuse struct DNNData here, actually the DNNData.data field is not needed.
    DNNReturnType (*get_input)(void *model, DNNData *input, const char *input_name);
    // Sets model input and output.
    // Should be called at least once before model execution.
    DNNReturnType (*set_input_output)(void *model, DNNData *input, const char *input_name, 
                    const char **output_names, uint32_t nb_output, uint32_t gpuid);
} DNNModel;

typedef struct DNNModule{
    // Loads model and parameters from given file. Returns NULL if it is not possible.
    DNNModel *(*load_model)(const char *model_filename);
    // Executes model with specified input and output. Returns DNN_ERROR otherwise.
    DNNReturnType (*execute_model)(const DNNModel *model, DNNData *outputs, uint32_t nb_output);
    // Frees memory allocated for model.
    void (*free_model)(DNNModel **model);
} DNNModule;


typedef struct LVPDnnContext {    

    DNNBackendType backend_type;  //default tensorflow
    int     filter_type;          //classification, rect detection ...
    char    *model_filename;    
    char    *model_inputname;
    char    *model_outputname;
    int     sample_rate;
    int     gpuid;
    float   valid_threshold;
    char    log_filename[MAXPATH];

    DNNModule   *dnn_module;
    DNNModel    *model;

    // internal context
    DNNData input;
    DNNData output;
    
    struct SwsContext   *sws_rgb_scale;
    struct SwsContext   *sws_gray8_to_grayf32;

	struct AVFrame		*readframe;
    struct AVFrame      *swscaleframe;
    struct AVFrame      *swframeforHW;

	//Video decode
	AVFormatContext 	*input_ctx;
	int 				video_stream;
	AVCodecContext 		*decoder_ctx;
	AVCodec 			*decoder;

	//for HW accelerate
	enum AVHWDeviceType type;
	AVBufferRef 		*hw_device_ctx;
	enum AVPixelFormat 	hw_pix_fmt;	
	
	// for log file
    FILE                *logfile;
    int                 framenum;

} LVPDnnContext;

int     lpms_dnninit(char* fmodelpath, char* input, char* output, int samplerate, float fthreshold);
void  	lpms_dnnfree();
int  	lpms_dnnexecute(char* ivpath, int  flagHW, int  flagclass,float  tinteval, float* porob);

//added for multiple model
LVPDnnContext*  lpms_dnnnew();
int  lpms_dnninitwithctx(LVPDnnContext* ctx, char* fmodelpath, char* input, char* output, int samplerate, float fthreshold, int gpuid);
int  lpms_dnnexecutewithctx(LVPDnnContext *context, char* ivpath, int flagHW, float tinteval, int* classid, float* porob);
void lpms_dnnstop(LVPDnnContext* context);

Vinfo*  lpms_vinfonew();
int     lpms_getvideoinfo(char* ivpath, Vinfo* vinfo);

#endif
#endif // _LPMS_FFMPEG_H_
