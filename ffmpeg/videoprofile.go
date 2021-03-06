package ffmpeg

import (
	"strconv"
	"strings"

	"github.com/golang/glog"
	"github.com/oscar-davids/lpmsdemo/m3u8"
)

const (
	DnnClassify = iota
	DnnYolo
	DnnMasking
)

const (
	Subtitle = iota
	MpegMetadata
	HLSMetadata
)

type DetectorProfile struct {
	Dnntype    int
	SampleRate uint
	ModelPath  string
	Threshold  float32
	Input      string
	Output     string
	Gpuid      int
	ClassID    int // now need?
	MetaMode   int // 0: subtitle(default), 1: ffmpeg metadata, 2: hls timed metadata(Reservation)
	Interval   float32
	ClassName  []string
}

//Standard Profiles:
//1080p60fps: 9000kbps
//1080p30fps: 6000kbps
//720p60fps: 6000kbps
//720p30fps: 4000kbps
//480p30fps: 2000kbps
//360p30fps: 1000kbps
//240p30fps: 700kbps
//144p30fps: 400kbps
type VideoProfile struct {
	Name        string
	Bitrate     string
	Framerate   uint
	Resolution  string
	AspectRatio string

	Detector DetectorProfile
}
//dnnmodel description
//tadmodel.pb : adult detection classification model
//tafmodel.pb : adult & football match detection classification model
//tviomodel.pb : violence detection classification model
//tyolov3model.pb : yolo v3 object detection model

//Some sample video profiles
var (
	P720p60fps16x9 = VideoProfile{Name: "P720p60fps16x9", Bitrate: "6000k", Framerate: 60, AspectRatio: "16:9", Resolution: "1280x720"}
	P720p30fps16x9 = VideoProfile{Name: "P720p30fps16x9", Bitrate: "4000k", Framerate: 30, AspectRatio: "16:9", Resolution: "1280x720"}
	P720p25fps16x9 = VideoProfile{Name: "P720p25fps16x9", Bitrate: "3500k", Framerate: 25, AspectRatio: "16:9", Resolution: "1280x720"}
	P720p30fps4x3  = VideoProfile{Name: "P720p30fps4x3", Bitrate: "3500k", Framerate: 30, AspectRatio: "4:3", Resolution: "960x720"}
	P576p30fps16x9 = VideoProfile{Name: "P576p30fps16x9", Bitrate: "1500k", Framerate: 30, AspectRatio: "16:9", Resolution: "1024x576"}
	P576p25fps16x9 = VideoProfile{Name: "P576p25fps16x9", Bitrate: "1500k", Framerate: 25, AspectRatio: "16:9", Resolution: "1024x576"}
	P360p30fps16x9 = VideoProfile{Name: "P360p30fps16x9", Bitrate: "1200k", Framerate: 30, AspectRatio: "16:9", Resolution: "640x360"}
	P360p25fps16x9 = VideoProfile{Name: "P360p25fps16x9", Bitrate: "1000k", Framerate: 25, AspectRatio: "16:9", Resolution: "640x360"}
	P360p30fps4x3  = VideoProfile{Name: "P360p30fps4x3", Bitrate: "1000k", Framerate: 30, AspectRatio: "4:3", Resolution: "480x360"}
	P240p30fps16x9 = VideoProfile{Name: "P240p30fps16x9", Bitrate: "600k", Framerate: 30, AspectRatio: "16:9", Resolution: "426x240"}
	P240p25fps16x9 = VideoProfile{Name: "P240p25fps16x9", Bitrate: "600k", Framerate: 25, AspectRatio: "16:9", Resolution: "426x240"}
	P240p30fps4x3  = VideoProfile{Name: "P240p30fps4x3", Bitrate: "600k", Framerate: 30, AspectRatio: "4:3", Resolution: "320x240"}
	P144p30fps16x9 = VideoProfile{Name: "P144p30fps16x9", Bitrate: "400k", Framerate: 30, AspectRatio: "16:9", Resolution: "256x144"}
	P144p25fps16x9 = VideoProfile{Name: "P144p25fps16x9", Bitrate: "400k", Framerate: 25, AspectRatio: "16:9", Resolution: "256x144"}
	PDnnDetector   = VideoProfile{Name: "PDnnDetector", Bitrate: "400k", Framerate: 20, AspectRatio: "1:1", Resolution: "224x224",
		Detector: DetectorProfile{SampleRate: 30, ModelPath: "tafmodel.pb", Threshold: 0.8, Input: "input_1", Output: "reshape_3/Reshape",
			ClassID: 0, MetaMode: 0, ClassName: []string{"adult", "football match"}}}
	PDnnVioFilter = VideoProfile{Name: "PDnnVioFilter", Bitrate: "400k", Framerate: 20, AspectRatio: "1:1", Resolution: "224x224",
		Detector: DetectorProfile{SampleRate: 30, ModelPath: "tviomodel.pb", Threshold: 0.8, Input: "input_1", Output: "reshape_3/Reshape",
			ClassID: 0, MetaMode: 0, ClassName: []string{"violence"}}}
	PDnnOtherFilter = VideoProfile{Name: "PDnnOtherFilter", Bitrate: "400k", Framerate: 20, AspectRatio: "1:1", Resolution: "224x224",
		Detector: DetectorProfile{SampleRate: 30, ModelPath: "tadmodel.pb", Threshold: 0.8, Input: "input_1", Output: "reshape_3/Reshape",
			ClassID: 0, MetaMode: 0, ClassName: []string{"adult", "football match"}}}
	PDnnYoloFilter = VideoProfile{Name: "PDnnYoloFilter", Bitrate: "400k", Framerate: 20, AspectRatio: "1:1", Resolution: "416x416",
		Detector: DetectorProfile{Dnntype: DnnYolo, SampleRate: 30, ModelPath: "tyolov3model.pb", Threshold: 0.6, Input: "inputs", Output: "output_boxes",
			ClassID: 0, MetaMode: 0, ClassName: []string{"person", "bicycle", "car", "motorbike", "aeroplane",
				"bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
				"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
				"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
				"carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
				"keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"}}}
)

var VideoProfileLookup = map[string]VideoProfile{
	"P720p60fps16x9":  P720p60fps16x9,
	"P720p30fps16x9":  P720p30fps16x9,
	"P720p25fps16x9":  P720p25fps16x9,
	"P720p30fps4x3":   P720p30fps4x3,
	"P576p30fps16x9":  P576p30fps16x9,
	"P576p25fps16x9":  P576p25fps16x9,
	"P360p30fps16x9":  P360p30fps16x9,
	"P360p25fps16x9":  P360p25fps16x9,
	"P360p30fps4x3":   P360p30fps4x3,
	"P240p30fps16x9":  P240p30fps16x9,
	"P240p25fps16x9":  P240p25fps16x9,
	"P240p30fps4x3":   P240p30fps4x3,
	"P144p30fps16x9":  P144p30fps16x9,
	"PDnnDetector":    PDnnDetector,
	"PDnnVioFilter":   PDnnVioFilter,
	"PDnnOtherFilter": PDnnOtherFilter,
	"PDnnYoloFilter":  PDnnYoloFilter,
}

func VideoProfileResolution(p VideoProfile) (int, int, error) {
	res := strings.Split(p.Resolution, "x")
	if len(res) < 2 {
		return 0, 0, ErrTranscoderRes
	}
	w, err := strconv.Atoi(res[0])
	if err != nil {
		return 0, 0, err
	}
	h, err := strconv.Atoi(res[1])
	if err != nil {
		return 0, 0, err
	}
	return w, h, nil
}

func VideoProfileToVariantParams(p VideoProfile) m3u8.VariantParams {
	r := p.Resolution
	r = strings.Replace(r, ":", "x", 1)

	bw := p.Bitrate
	bw = strings.Replace(bw, "k", "000", 1)
	b, err := strconv.ParseUint(bw, 10, 32)
	if err != nil {
		glog.Errorf("Error converting %v to variant params: %v", bw, err)
	}
	return m3u8.VariantParams{Bandwidth: uint32(b), Resolution: r}
}

type ByName []VideoProfile

func (a ByName) Len() int      { return len(a) }
func (a ByName) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ByName) Less(i, j int) bool {
	return a[i].Name > a[j].Name
} //Want to sort in reverse

// func bitrateStrToInt(bitrateStr string) int {
// 	intstr := strings.Replace(bitrateStr, "k", "000", 1)
// 	res, _ := strconv.Atoi(intstr)
// 	return res
// }
