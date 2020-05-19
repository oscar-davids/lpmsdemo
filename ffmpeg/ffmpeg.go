package ffmpeg

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/golang/glog"
)

// #cgo pkg-config: libavformat libavfilter libavcodec libavutil libswscale gnutls
// #cgo LDFLAGS: -ltensorflow
// #include <stdlib.h>
// #include "lpms_ffmpeg.h"
import "C"

var ErrTranscoderRes = errors.New("TranscoderInvalidResolution")
var ErrTranscoderHw = errors.New("TranscoderInvalidHardware")
var ErrTranscoderInp = errors.New("TranscoderInvalidInput")
var ErrTranscoderStp = errors.New("TranscoderStopped")

type Acceleration int

const (
	Software Acceleration = iota
	Nvidia
	Amd
)

type ComponentOptions struct {
	Name string
	Opts map[string]string
}

type Transcoder struct {
	handle  *C.struct_transcode_thread
	stopped bool
	mu      *sync.Mutex
}

type TranscodeOptionsIn struct {
	Fname      string
	Accel      Acceleration
	Device     string
	ParallelID int
}

type TranscodeOptions struct {
	Oname   string
	Profile VideoProfile
	Accel   Acceleration
	Device  string

	Muxer        ComponentOptions
	VideoEncoder ComponentOptions
	AudioEncoder ComponentOptions
}

type MediaInfo struct {
	Frames int
	Pixels int64
}

type TranscodeResults struct {
	Decoded    MediaInfo
	Encoded    []MediaInfo
	DetectProb float32
	Contents   string
}

//for multiple model
type VideoInfo struct {
	Vinfo *C.Vinfo
	init  bool
}

type DnnFilter struct {
	handle  *C.LVPDnnContext
	initdnn bool
	stopped bool
	mu      *sync.Mutex
	dnncfg  VideoProfile
}

type DnnSet struct {
	streamId string
	gpuid    uint
	filters  []DnnFilter
}

type GpuStatus struct {
	usage     int
	streamIds []string
}

var initengine bool = false
var dnnfilters []DnnFilter
var dnnsets []DnnSet //now used
var gpuparallel int = 0
var gpunum int = 0
var gpuusage []GpuStatus
var usednnCengine bool = false
var ftimeinterval float32 = 0.0

//in the future
//var dnnMatrix [][]DnnSet

func RTMPToHLS(localRTMPUrl string, outM3U8 string, tmpl string, seglen_secs string, seg_start int) error {
	inp := C.CString(localRTMPUrl)
	outp := C.CString(outM3U8)
	ts_tmpl := C.CString(tmpl)
	seglen := C.CString(seglen_secs)
	segstart := C.CString(fmt.Sprintf("%v", seg_start))
	ret := int(C.lpms_rtmp2hls(inp, outp, ts_tmpl, seglen, segstart))
	C.free(unsafe.Pointer(inp))
	C.free(unsafe.Pointer(outp))
	C.free(unsafe.Pointer(ts_tmpl))
	C.free(unsafe.Pointer(seglen))
	C.free(unsafe.Pointer(segstart))
	if 0 != ret {
		glog.Infof("RTMP2HLS Transmux Return : %v\n", Strerror(ret))
		return ErrorMap[ret]
	}
	return nil
}

//call from subscriber
func Transcode(input string, workDir string, pid int, gid int, ps []VideoProfile) (string, error) {
	sdev := fmt.Sprintf("%d", gid)
	opts := make([]TranscodeOptions, len(ps))
	for i, param := range ps {
		oname := path.Join(workDir, fmt.Sprintf("out%v%v", i, filepath.Base(input)))
		//oname = ".tmp/" + fmt.Sprintf("out%v%v", i, filepath.Base(input))
		opt := TranscodeOptions{
			Oname:   oname,
			Profile: param,
			Accel:   Nvidia,
			Device:  sdev,
		}
		opts[i] = opt
	}
	inopts := &TranscodeOptionsIn{
		Fname:      input,
		Accel:      Nvidia,
		Device:     sdev,
		ParallelID: pid,
	}
	return TranscodeAndDetection(inopts, opts)
}

func TranscodeAndDetection(input *TranscodeOptionsIn, ps []TranscodeOptions) (string, error) {
	res, err := Transcode3(input, ps)
	return res.Contents, err
}

func newAVOpts(opts map[string]string) *C.AVDictionary {
	var dict *C.AVDictionary
	for key, value := range opts {
		k := C.CString(key)
		v := C.CString(value)
		defer C.free(unsafe.Pointer(k))
		defer C.free(unsafe.Pointer(v))
		C.av_dict_set(&dict, k, v, 0)
	}
	return dict
}

// return encoding specific options for the given accel
func configAccel(inAcc, outAcc Acceleration, inDev, outDev string) (string, string, error) {
	switch inAcc {
	case Software:
		switch outAcc {
		case Software:
			return "libx264", "scale", nil
		case Nvidia:
			upload := "hwupload_cuda"
			if outDev != "" {
				upload = upload + "=device=" + outDev
			}
			return "h264_nvenc", upload + ",scale_cuda", nil
		}
	case Nvidia:
		switch outAcc {
		case Software:
			return "libx264", "scale_cuda", nil
		case Nvidia:
			// If we encode on a different device from decode then need to transfer
			if outDev != "" && outDev != inDev {
				return "", "", ErrTranscoderInp // XXX not allowed
			}
			return "h264_nvenc", "scale_cuda", nil
		}
	}
	return "", "", ErrTranscoderHw
}
func accelDeviceType(accel Acceleration) (C.enum_AVHWDeviceType, error) {
	switch accel {
	case Software:
		return C.AV_HWDEVICE_TYPE_NONE, nil
	case Nvidia:
		return C.AV_HWDEVICE_TYPE_CUDA, nil

	}
	return C.AV_HWDEVICE_TYPE_NONE, ErrTranscoderHw
}

func Transcode2(input *TranscodeOptionsIn, ps []TranscodeOptions) error {
	_, err := Transcode3(input, ps)
	return err
}

func Transcode3(input *TranscodeOptionsIn, ps []TranscodeOptions) (*TranscodeResults, error) {
	t := NewTranscoder()
	defer t.StopTranscoder()
	return t.Transcode(input, ps)
}

func Transcode4(input *TranscodeOptionsIn, ps []TranscodeOptions) (*TranscodeResults, error) {
	t := NewTranscoder()
	defer t.StopTranscoder()
	return t.Transcode(input, ps)
}

//var gdetector = NewTranscoder()
//defer gdetector.StopTranscoder()

func (t *Transcoder) Detector(input *TranscodeOptionsIn, p TranscodeOptions) (float32, error) {

	if input == nil {
		return 0.0, ErrTranscoderInp
	}
	hw_type, err := accelDeviceType(input.Accel)
	if err != nil {
		return 0.0, err
	}

	fname := C.CString(input.Fname)
	defer C.free(unsafe.Pointer(fname))

	oname := C.CString(p.Oname)
	defer C.free(unsafe.Pointer(oname))

	param := p.Profile

	dcfg := param.Detector
	detecttemp := p.Oname + "_cdump.srt"
	filters := fmt.Sprintf("lvpdnn=model=%s:input=%s:output=%s:sample=%d:threshold=%f:log=%s",
		dcfg.ModelPath, dcfg.Input, dcfg.Output, dcfg.SampleRate, dcfg.Threshold, detecttemp)

	samplerate := param.Detector.SampleRate

	muxOpts := C.component_opts{
		name: C.CString(p.Muxer.Name),
		opts: newAVOpts(p.VideoEncoder.Opts),
	}
	vidOpts := C.component_opts{
		name: C.CString(p.VideoEncoder.Name),
		opts: newAVOpts(p.VideoEncoder.Opts),
	}
	audioOpts := C.component_opts{
		name: C.CString("aac"),
		opts: newAVOpts(p.AudioEncoder.Opts),
	}

	vfilt := C.CString(filters)
	defer C.free(unsafe.Pointer(muxOpts.name))
	defer C.free(unsafe.Pointer(vidOpts.name))
	defer C.free(unsafe.Pointer(audioOpts.name))
	defer C.free(unsafe.Pointer(vfilt))

	var fps C.AVRational
	if param.Framerate > 0 {
		fps = C.AVRational{num: C.int(param.Framerate), den: 1}
	}
	cparam := C.output_params{fname: oname, fps: fps,
		w: C.int(224), h: C.int(224), bitrate: C.int(40000),
		muxer: muxOpts, audio: audioOpts, video: vidOpts, vfilters: vfilt}

	defer func(cparam *C.output_params) {
		// Work around the ownership rules:
		// ffmpeg normally takes ownership of the following AVDictionary options
		// However, if we don't pass these opts to ffmpeg, then we need to free
		if cparam.muxer.opts != nil {
			C.av_dict_free(&cparam.muxer.opts)
		}
		if cparam.audio.opts != nil {
			C.av_dict_free(&cparam.audio.opts)
		}
		if cparam.video.opts != nil {
			C.av_dict_free(&cparam.video.opts)
		}
	}(&cparam)

	var device *C.char
	if input.Device != "" {
		device = C.CString(input.Device)
		defer C.free(unsafe.Pointer(device))
	}
	inp := &C.input_params{fname: fname, hw_type: hw_type, device: device,
		handle: t.handle}
	results := &C.output_results{}
	decoded := &C.output_results{}
	var (
		paramsPointer *C.output_params
		//resultsPointer *C.output_results
	)
	paramsPointer = (*C.output_params)(&cparam)

	ret := int(C.lpms_transcode(inp, paramsPointer, results, C.int(1), decoded))

	if 0 != ret {
		glog.Error("Transcoder Return : ", ErrorMap[ret])
		return 0.0, ErrorMap[ret]
	}
	var fconfidence float32 = 0.0
	if detecttemp != "" && samplerate > 0 {

		file, err := os.Open(detecttemp)
		if err != nil {
			glog.Infof("Can not open Detection dump file : %s\n", detecttemp)
		}

		defer file.Close()

		fileScanner := bufio.NewScanner(file)
		lineCount := 0
		for fileScanner.Scan() {
			lineCount++
		}

		checkedframe := int(decoded.frames) / int(samplerate)
		fconfidence = float32(lineCount) / float32(checkedframe)

		os.Remove(detecttemp)
	}

	return fconfidence, nil
}

func (t *Transcoder) ExecuteSetFilter(infname string, Accel Acceleration) (subtfname string, srtmetadata string, fconfidence float32) {

	subtfname = ""
	srtmetadata = ""
	fconfidence = 0.0

	if len(dnnfilters) > 0 {
		bmetadata := false
		bcontent := false
		if dnnfilters[0].dnncfg.Detector.MetaMode >= MpegMetadata {
			bmetadata = true
		}
		if bmetadata == true { //not sub title mode
			for i, ft := range dnnfilters {
				clsid, confidence := ft.ExecuteDnnFilter(infname, Accel)
				if i == 0 {
					fconfidence = confidence //for tranncoding sample
				}
				if confidence >= ft.dnncfg.Detector.Threshold && clsid >= 0 && clsid < len(ft.dnncfg.Detector.ClassName) {
					if len(srtmetadata) > 0 {
						srtmetadata += ", "
					}
					srtmetadata += ft.dnncfg.Detector.ClassName[clsid]
				}
			}
		} else { //subtitle mode
			subtfname = "subtitle.srt"
			srtfile, err := os.Create(subtfname)
			if err == nil {
				//glog.Infof("Can not open subtitle.srt file %v\n", err)
				fmt.Fprint(srtfile, 1, "\n", "00:00:00.0 --> 00:10:00.0", "\n")
			}

			for i, ft := range dnnfilters {
				clsid, confidence := ft.ExecuteDnnFilter(infname, Accel)
				if i == 0 {
					fconfidence = confidence //for tranncoding sample
				}
				if confidence >= ft.dnncfg.Detector.Threshold && clsid >= 0 && clsid < len(ft.dnncfg.Detector.ClassName) && err == nil {
					bcontent = true
					fmt.Fprint(srtfile, "content: ", ft.dnncfg.Detector.ClassName[clsid], "!\n")
				}
			}

			if bcontent == false {
				subtfname = ""
			}
			srtfile.Close()
		}
	}
	return subtfname, srtmetadata, fconfidence
}
func (t *Transcoder) Transcode(input *TranscodeOptionsIn, psin []TranscodeOptions) (*TranscodeResults, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.stopped || t.handle == nil {
		return nil, ErrTranscoderStp
	}
	if input == nil {
		return nil, ErrTranscoderInp
	}
	hw_type, err := accelDeviceType(input.Accel)
	if err != nil {
		return nil, err
	}
	fname := C.CString(input.Fname)
	defer C.free(unsafe.Pointer(fname))
	//var detecttemp string
	//var samplerate uint

	//scane dnn dector
	ps := []TranscodeOptions{}
	pdnn := TranscodeOptions{}

	for _, p := range psin {
		if p.Profile.Detector.SampleRate > 0 {
			pdnn.Profile = p.Profile
			pdnn.Muxer.Name = "null"
			pdnn.VideoEncoder.Name = "rawvideo"
		} else {
			ps = append(ps, p)
		}
	}

	//make srt format file
	var subtfname string = ""
	var srtmetadata string = ""
	var fconfidence float32 = 0.0

	if usednnCengine == false {
		if gpuparallel > 0 {
			glog.Infof("Parallel ID / Parallel Count: %v/%v\n", input.ParallelID, gpuparallel)

			if input.ParallelID >= 0 && input.ParallelID < gpuparallel {
				subtfname, srtmetadata = dnnsets[input.ParallelID].ExecuteSetDnnFilter(input.Fname, input.Accel)
			}

		} else {
			subtfname, srtmetadata, fconfidence = t.ExecuteSetFilter(input.Fname, input.Accel)
		}
	}

	params := make([]C.output_params, len(ps))
	for i, p := range ps {
		oname := C.CString(p.Oname)
		defer C.free(unsafe.Pointer(oname))

		param := p.Profile
		w, h, err := VideoProfileResolution(param)
		if err != nil {
			if "drop" != p.VideoEncoder.Name && "copy" != p.VideoEncoder.Name {
				return nil, err
			}
		}
		br := strings.Replace(param.Bitrate, "k", "000", 1)
		bitrate, err := strconv.Atoi(br)
		if err != nil {
			if "drop" != p.VideoEncoder.Name && "copy" != p.VideoEncoder.Name {
				return nil, err
			}
		}
		encoder, scale_filter := p.VideoEncoder.Name, "scale"
		if encoder == "" {
			encoder, scale_filter, err = configAccel(input.Accel, p.Accel, input.Device, p.Device)
			if err != nil {
				return nil, err
			}
		}
		// preserve aspect ratio along the larger dimension when rescaling
		var filters string
		if param.Framerate > 0 {
			filters = fmt.Sprintf("fps=%d/1,", param.Framerate)
		}
		if usednnCengine == false && len(subtfname) > 0 {
			if input.Accel == Software {
				filters += fmt.Sprintf("subtitles=%v,", subtfname)
			} else {
				filters += fmt.Sprintf("hwdownload,format=nv12,subtitles=%v,", subtfname)
			}

			if p.Accel == Nvidia {
				filters += fmt.Sprintf("hwupload_cuda,")
			}
			//filters += "subtitles=subtitle.srt,"
		}
		filters += fmt.Sprintf("%s='w=if(gte(iw,ih),%d,-2):h=if(lt(iw,ih),%d,-2)'", scale_filter, w, h)
		if input.Accel != Software && p.Accel == Software {
			// needed for hw dec -> hw rescale -> sw enc
			filters = filters + ",hwdownload,format=nv12"
		}
		muxOpts := C.component_opts{
			opts: newAVOpts(p.Muxer.Opts), // don't free this bc of avformat_write_header API
		}
		if p.Muxer.Name != "" {
			muxOpts.name = C.CString(p.Muxer.Name)
			defer C.free(unsafe.Pointer(muxOpts.name))
		}
		// Set some default encoding options
		if len(p.VideoEncoder.Name) <= 0 && len(p.VideoEncoder.Opts) <= 0 {
			p.VideoEncoder.Opts = map[string]string{
				"forced-idr": "1",
			}
		}
		vidOpts := C.component_opts{
			name: C.CString(encoder),
			opts: newAVOpts(p.VideoEncoder.Opts),
		}
		audioEncoder := p.AudioEncoder.Name
		if audioEncoder == "" {
			audioEncoder = "aac"
		}
		audioOpts := C.component_opts{
			name: C.CString(audioEncoder),
			opts: newAVOpts(p.AudioEncoder.Opts),
		}
		vfilt := C.CString(filters)
		defer C.free(unsafe.Pointer(vidOpts.name))
		defer C.free(unsafe.Pointer(audioOpts.name))
		defer C.free(unsafe.Pointer(vfilt))
		var fps C.AVRational
		if param.Framerate > 0 {
			fps = C.AVRational{num: C.int(param.Framerate), den: 1}
		}
		params[i] = C.output_params{fname: oname, fps: fps,
			w: C.int(w), h: C.int(h), bitrate: C.int(bitrate),
			muxer: muxOpts, audio: audioOpts, video: vidOpts, vfilters: vfilt}
		defer func(param *C.output_params) {
			// Work around the ownership rules:
			// ffmpeg normally takes ownership of the following AVDictionary options
			// However, if we don't pass these opts to ffmpeg, then we need to free
			if param.muxer.opts != nil {
				C.av_dict_free(&param.muxer.opts)
			}
			if param.audio.opts != nil {
				C.av_dict_free(&param.audio.opts)
			}
			if param.video.opts != nil {
				C.av_dict_free(&param.video.opts)
			}
		}(&params[i])
	}
	var device *C.char
	if input.Device != "" {
		device = C.CString(input.Device)
		defer C.free(unsafe.Pointer(device))
	}
	var smetadata *C.char = nil
	if usednnCengine == false && len(srtmetadata) > 0 { //ffmpeg meta data mode
		//glog.Infof("DnnFilter metadata: %v", srtmetadata)
		smetadata = C.CString(srtmetadata)
		defer C.free(unsafe.Pointer(smetadata))
	}

	inp := &C.input_params{fname: fname, hw_type: hw_type, device: device, metadata: smetadata,
		handle: t.handle, ftimeinterval: C.float(ftimeinterval)}

	results := make([]C.output_results, len(ps))
	decoded := &C.output_results{}
	var (
		paramsPointer  *C.output_params
		resultsPointer *C.output_results
	)
	if len(params) > 0 {
		paramsPointer = (*C.output_params)(&params[0])
		resultsPointer = (*C.output_results)(&results[0])
	}
	ret := int(C.lpms_transcode(inp, paramsPointer, resultsPointer, C.int(len(params)), decoded))
	if 0 != ret {
		glog.Error("Transcoder Return : ", ErrorMap[ret])
		return nil, ErrorMap[ret]
	}
	tr := make([]MediaInfo, len(ps))
	for i, r := range results {
		tr[i] = MediaInfo{
			Frames: int(r.frames),
			Pixels: int64(r.pixels),
		}
	}
	dec := MediaInfo{
		Frames: int(decoded.frames),
		Pixels: int64(decoded.pixels),
	}

	if gpuparallel > 0 && dnnsets[0].filters[0].dnncfg.Detector.MetaMode == HLSMetadata {
		return &TranscodeResults{Encoded: tr, Decoded: dec, DetectProb: fconfidence, Contents: srtmetadata}, nil
	} else if len(dnnfilters) > 0 && dnnfilters[0].dnncfg.Detector.MetaMode == HLSMetadata {
		return &TranscodeResults{Encoded: tr, Decoded: dec, DetectProb: fconfidence, Contents: srtmetadata}, nil
	} else {
		return &TranscodeResults{Encoded: tr, Decoded: dec, DetectProb: fconfidence}, nil
	}
}

func NewTranscoder() *Transcoder {
	return &Transcoder{
		handle: C.lpms_transcode_new(),
		mu:     &sync.Mutex{},
	}
}

func (t *Transcoder) StopTranscoder() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.stopped {
		return
	}
	C.lpms_transcode_stop(t.handle)
	t.handle = nil // prevent accidental reuse
	t.stopped = true
}

func InitFFmpeg() {
	C.lpms_init()
}

// get video info implementation
func NewDnnVinfo() *VideoInfo {
	return &VideoInfo{
		Vinfo: C.lpms_vinfonew(),
		init:  false,
	}
}
func (t *VideoInfo) GetVideoInfo(infname string) string {
	fname := C.CString(infname)
	defer C.free(unsafe.Pointer(fname))
	C.lpms_getvideoinfo(fname, t.Vinfo)

	srtret := fmt.Sprintf("%vX%v @ %vfps %v sec", int(t.Vinfo.width), int(t.Vinfo.height), float32(t.Vinfo.fps), float32(t.Vinfo.duration))

	return srtret
}
func (t *VideoInfo) DeleteDnnVinfo() {
	C.free(unsafe.Pointer(t.Vinfo))
	t.Vinfo = nil
}

func InitDnnEngine(dnncfg VideoProfile) {
	if initengine == false {

		model := C.CString(dnncfg.Detector.ModelPath)
		defer C.free(unsafe.Pointer(model))
		Input := C.CString(dnncfg.Detector.Input)
		defer C.free(unsafe.Pointer(Input))
		Output := C.CString(dnncfg.Detector.Output)
		defer C.free(unsafe.Pointer(Output))
		nsample := int(dnncfg.Detector.SampleRate)
		threshold := dnncfg.Detector.Threshold

		C.lpms_dnninit(model, Input, Output, C.int(nsample), C.float(threshold))
		initengine = true
	}
}
func ReleaseDnnEngine() {
	if initengine == true {
		C.lpms_dnnfree()
	}
}
func NewDnnFilter() *DnnFilter {
	return &DnnFilter{
		handle:  C.lpms_dnnnew(),
		initdnn: false,
		stopped: true,
		mu:      &sync.Mutex{},
	}
}
func (t *DnnFilter) InitDnnFilter(dnncfg VideoProfile) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.initdnn {
		return true
	}
	model := C.CString(dnncfg.Detector.ModelPath)
	defer C.free(unsafe.Pointer(model))
	Input := C.CString(dnncfg.Detector.Input)
	defer C.free(unsafe.Pointer(Input))
	Output := C.CString(dnncfg.Detector.Output)
	defer C.free(unsafe.Pointer(Output))
	nsample := int(dnncfg.Detector.SampleRate)
	threshold := dnncfg.Detector.Threshold

	gpuid := int(dnncfg.Detector.Gpuid)

	res := C.lpms_dnninitwithctx(t.handle, model, Input, Output, C.int(nsample), C.float(threshold), C.int(gpuid))
	if res == 0 {
		t.initdnn = true
		t.stopped = false
		return true
	} else {
		return false
	}
}
func (t *DnnFilter) ExecuteDnnFilter(infname string, Accel Acceleration) (int, float32) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.stopped || t.initdnn == false {
		return -1, 0.0
	}
	flagHW := 0
	if Accel == Nvidia {
		flagHW = 1
	}

	fname := C.CString(infname)
	defer C.free(unsafe.Pointer(fname))

	var fconfidence float32 = 0.0
	var classid int = -1
	prob := C.float(fconfidence)
	//flagclass := C.int(t.dnncfg.Detector.ClassID)
	flagclass := C.int(classid)
	tinterval := C.float(t.dnncfg.Detector.Interval)
	C.lpms_dnnexecutewithctx(t.handle, fname, C.int(flagHW), tinterval, &flagclass, &prob)
	fconfidence = float32(prob)
	classid = int(flagclass)

	return classid, fconfidence
}
func (t *DnnFilter) StopDnnFilter() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.stopped {
		return
	}
	C.lpms_dnnstop(t.handle)
	t.handle = nil
	t.stopped = true
	t.initdnn = false
}

func (t *DnnSet) ExecuteSetDnnFilter(infname string, Accel Acceleration) (subtfname string, srtmetadata string) {

	bmetadata := false
	subtfname = ""
	srtmetadata = ""

	if len(t.filters) > 0 {
		bcontent := false
		if t.filters[0].dnncfg.Detector.MetaMode >= MpegMetadata {
			bmetadata = true
		}
		if bmetadata == true { //not sub title mode
			for _, ft := range t.filters {
				clsid, confidence := ft.ExecuteDnnFilter(infname, Accel)
				if confidence >= ft.dnncfg.Detector.Threshold && clsid >= 0 && clsid < len(ft.dnncfg.Detector.ClassName) {
					if len(srtmetadata) > 0 {
						srtmetadata += ", "
					}
					srtmetadata += ft.dnncfg.Detector.ClassName[clsid]
				}
			}
		} else { //subtitle mode
			subtfname = t.streamId + ".srt"
			srtfile, err := os.Create(subtfname)
			if err == nil {
				//glog.Infof("Can not open subtitle.srt file %v\n", err)
				fmt.Fprint(srtfile, 1, "\n", "00:00:00.0 --> 00:10:00.0", "\n")
			}

			for _, ft := range t.filters {
				clsid, confidence := ft.ExecuteDnnFilter(infname, Accel)
				if confidence >= ft.dnncfg.Detector.Threshold && clsid >= 0 && clsid < len(ft.dnncfg.Detector.ClassName) && err == nil {
					bcontent = true
					fmt.Fprint(srtfile, "content: ", ft.dnncfg.Detector.ClassName[clsid], "!\n")
				}
			}

			if bcontent == false {
				subtfname = ""
			}
			srtfile.Close()
		}
	}

	return subtfname, srtmetadata
}
func (t *DnnSet) StopSetDnnFilter() {
	t.streamId = ""
}
func (t *DnnSet) ReleaseSetDnnFilter() {

	for _, filter := range t.filters {
		filter.StopDnnFilter()
	}
	t.streamId = ""
}

//gloabal API

//for multiple model
//variable for vertical extention
//var dnnMatrix [][]DnnSet
//var dnnsets []DnnSet //now used
//var gpunum int = 0
//var gpuparallel int = 1

func SetAvailableGpuNum(ngpu int) int {
	gpunum = ngpu
	if gpunum > 0 {
		gpuusage = make([]GpuStatus, gpunum)

		for i, _ := range gpuusage {
			gpuusage[i].usage = 0
		}
	}
	//set tensorflow device setting
	return gpunum
}
func GetAvailableGpuNum() int {
	return gpunum
}
func GetGpuIdx(sid string) int {
	gpuid := -1
	usemin := 1000
	for i, _ := range gpuusage {
		if gpuusage[i].usage == 0 {
			gpuid = i
			break
		}
		if usemin > gpuusage[i].usage {
			usemin = gpuusage[i].usage
			gpuid = i
		}
	}

	if gpuid != -1 {
		gpuusage[gpuid].usage++
		gpuusage[gpuid].streamIds = append(gpuusage[gpuid].streamIds, sid)
	}

	return gpuid
}
func RemoveGpuInx(sid string) {
	flagbreak := false
	for i, _ := range gpuusage {
		for j, _ := range gpuusage[i].streamIds {
			if gpuusage[i].streamIds[j] == sid {
				gpuusage[i].streamIds[j] = gpuusage[i].streamIds[len(gpuusage[i].streamIds)-1]
				gpuusage[i].streamIds = gpuusage[i].streamIds[:len(gpuusage[i].streamIds)-1]
				gpuusage[i].usage--
				flagbreak = true
				break
			}
		}
		if flagbreak == true {
			break
		}
	}
}
func RemoveGpuInxWithID(gid int) {
	if gid >= 0 && gid < len(gpuusage) {
		gpuusage[gid].usage--
		if gpuusage[gid].usage < 0 {
			gpuusage[gid].usage = 0
		}
	}

}

func SetParallelGpuNum(parallel int) int {
	gpuparallel = parallel
	if gpuparallel > 0 {
		dnnsets = make([]DnnSet, gpuparallel)
	}
	return gpuparallel
}
func GetParallelGpuNum() int {
	return gpuparallel
}

func AddParallelID(streamId string) int {
	pid := -1
	if gpuparallel > 0 {
		for i, _ := range dnnsets {
			if dnnsets[i].streamId == streamId {
				pid = i
				break
			}
		}
		if pid == -1 {
			for i, _ := range dnnsets {
				if dnnsets[i].streamId == "" {
					dnnsets[i].streamId = streamId
					pid = i
					break
				}
			}
		}
	} else {
		pid = 0
	}

	return pid
}
func RemoveParallelID(streamId string) {
	if gpuparallel > 0 {
		for i, _ := range dnnsets {
			if dnnsets[i].streamId == streamId {
				dnnsets[i].streamId = ""
				break
			}
		}
	}
}

func RegistryDnnEngine(dnncfg VideoProfile) {
	if gpuparallel > 0 {
		for i, _ := range dnnsets {
			dnnfilter := NewDnnFilter()
			gid := 0
			if gpunum > 0 {
				gid = i % gpunum
			}
			dnncfg.Detector.Gpuid = gid

			dnnfilter.dnncfg = dnncfg
			if dnnfilter.InitDnnFilter(dnncfg) == true {
				dnnsets[i].filters = append(dnnsets[i].filters, *dnnfilter)
				//glog.Infof("RegistryDnnEngine debug-1: %v", len(dnnsets[i].filters))
			}
		}
		//glog.Infof("RegistryDnnEngine debug-2 %v", len(dnnsets[0].filters))

	} else {
		dnnfilter := NewDnnFilter()
		dnnfilter.dnncfg = dnncfg
		if dnnfilter.InitDnnFilter(dnncfg) == true {
			dnnfilters = append(dnnfilters, *dnnfilter)
		}
	}
	if usednnCengine == true && ftimeinterval == 0.0 {
		ftimeinterval = dnncfg.Detector.Interval
	}
}
func RemoveAllDnnEngine() {
	if gpuparallel > 0 {
		for i, _ := range dnnsets {
			dnnsets[i].ReleaseSetDnnFilter()
		}

	} else {
		for _, filter := range dnnfilters {
			filter.StopDnnFilter()
		}
	}

}
