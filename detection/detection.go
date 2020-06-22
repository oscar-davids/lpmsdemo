package detection

import (
	"fmt"
	"math/rand"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/oscar-davids/lpmsdemo/transcoder"

	"github.com/golang/glog"
	"github.com/oscar-davids/lpmsdemo/ffmpeg"
	"github.com/oscar-davids/lpmsdemo/stream"
)

var HLSWaitTime = time.Second * 10

type exampleStream string

func (t *exampleStream) StreamID() string {
	return string(*t)
}

func randString(n int) string {
	rand.Seed(time.Now().UnixNano())
	x := make([]byte, n, n)
	for i := 0; i < len(x); i++ {
		x[i] = byte(rand.Uint32())
	}
	return fmt.Sprintf("%x", x)
}

func parseStreamID(reqPath string) string {
	var strmID string
	regex, _ := regexp.Compile("\\/stream\\/([[:alpha:]]|\\d)*")
	match := regex.FindString(reqPath)
	if match != "" {
		strmID = strings.Replace(match, "/stream/", "", -1)
	}
	return strmID
}

func getHLSSegmentName(url *url.URL) string {
	var segName string
	regex, _ := regexp.Compile("\\/stream\\/.*\\.ts")
	match := regex.FindString(url.Path)
	if match != "" {
		segName = strings.Replace(match, "/stream/", "", -1)
	}
	return segName
}
func getRTMPRequestName(url *url.URL) string {
	var reqName string
	regex, _ := regexp.Compile("\\/stream\\/.*")
	match := regex.FindString(url.Path)
	if match != "" {
		reqName = strings.Replace(match, "/stream/", "", -1)
	}
	return reqName
}

func validDnnfilters() []string {
	valids := make([]string, len(ffmpeg.VideoProfileLookup))
	for p, _ := range ffmpeg.VideoProfileLookup {
		if strings.Index(p, "PDnn") < 0 {
			continue
		}
		valids = append(valids, p)
	}
	return valids
}

func InitDnnEngine(profiles []ffmpeg.VideoProfile) {
	dir, err := os.Getwd()
	if err != nil {
		glog.Infof("Error getting work directory: %v", err)
	}
	glog.Infof("Settig working directory %v", fmt.Sprintf("%v/.tmp", dir))

	cengginflag := true
	ffmpeg.SetCengineFlag(cengginflag)
	ffmpeg.SetAvailableGpuNum(1)
	ffmpeg.SetParallelGpuNum(1)

	for _, ft := range profiles {
		ft.Detector.Interval = 1.0 //need profile from config
		ft.Detector.MetaMode = 2   //recommand HLS metatag mode
		glog.Infof("Registry DnnEngine: name: %v metadamode: %v", ft.Name, ft.Detector.MetaMode)
		ffmpeg.RegistryDnnEngine(ft)
	}

}

func ProcessSegment(seg *stream.HLSSegment, strmID string, profiles []ffmpeg.VideoProfile) ([][]byte, string, error) {
	workDir := ".tmp/"

	t := transcoder.NewFFMpegSegmentTranscoder(profiles, workDir)
	pid := ffmpeg.AddParallelID(strmID)

	if pid == -1 {
		glog.Errorf("Can not find transcoding pid with streamid: %v", strmID)
		// return nil
	}
	t.SetParallelID(pid)
	glog.Infof("Set PID, strmID: %v %v\n", pid, strmID)
	gpuid := ffmpeg.GetGpuIdx(strmID)
	if gpuid >= 0 {
		t.SetGpuID(gpuid)
	}

	getfile := workDir + seg.Name
	return t.Transcode(getfile)
}
