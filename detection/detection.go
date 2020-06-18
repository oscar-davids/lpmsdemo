package detection

import (
	"fmt"
	"io/ioutil"
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

func transcode(hlsStream stream.HLSVideoStream, flagclass int, tinterval float64) (func(*stream.HLSSegment, bool), error) {
	//Create Transcoder
	profiles := []ffmpeg.VideoProfile{
		ffmpeg.P720p25fps16x9,
		//ffmpeg.PDnnDetector,
		//ffmpeg.P240p30fps16x9,
		//ffmpeg.P576p30fps16x9,
	}
	workDir := ".tmp/"
	strmID := hlsStream.GetStreamID()
	t := transcoder.NewFFMpegSegmentTranscoder(profiles, workDir)
	//t.SetStreamID(strmID)
	pid := ffmpeg.AddParallelID(strmID)

	if pid == -1 {
		glog.Errorf("Can not find transcoding pid with streamid: %v", strmID)
		return nil, ffmpeg.ErrTranscoderStp
	}

	t.SetParallelID(pid)
	glog.Infof("Set PID, strmID: %v %v\n", pid, strmID)

	gpuid := ffmpeg.GetGpuIdx(strmID)
	if gpuid >= 0 {
		t.SetGpuID(gpuid)
	}
	precontents := ""
	//load temp warning video(warning.ts)
	warningbuff, err := ioutil.ReadFile("warning.ts")
	if err != nil {
		glog.Errorf("Can not read temp warning video(warning.ts): %v", err)
	}

	subscriber := func(seg *stream.HLSSegment, eof bool) {
		//If we get a new video segment for the original HLS stream, do the transcoding.
		//glog.Infof("Got seg: %v %v\n", seg.Name, hlsStream.GetStreamID())
		//getfile := ".tmp/" + seg.Name
		getfile := workDir + seg.Name
		//Transcode stream
		tData, contents, err := t.Transcode(getfile)
		if err != nil {
			glog.Errorf("Error transcoding: %v", err)
		} else {

			//Insert into HLS stream
			//for i, strmID := range strmIDs
			for i, p := range profiles {
				if strings.Index(p.Name, "PDnn") >= 0 {
					continue
				}
				//glog.Infof("Inserting transcoded seg %v into strm: %v", len(tData[i]), strmID)
				sName := fmt.Sprintf("%v_%v.ts", strmID, seg.SeqNo)
				PgDataTime := false
				PgDataEnd := false
				FgContents := 0 //0:Contents None, 1:ContentsStart, 2:ContentsContinue, 3:ContentsEnd

				if len(precontents) == 0 && len(contents) == 0 { //normal
					//write normal mode
					FgContents = stream.ContentsNone
				} else if len(precontents) == 0 && len(contents) > 0 { //started other contents
					//write EXT-X-PROGRAM-DATE-TIME start
					//write EXT-X-DATERANGE start
					PgDataTime = true
					FgContents = stream.ContentsStart

				} else if len(precontents) > 0 && len(contents) > 0 { //continue other contents
					//write EXT-X-DATERANGE continue
					FgContents = stream.ContentsContinue

				} else if len(precontents) > 0 && len(contents) == 0 { //ended other contents
					//write EXT-X-DISCONTINUITY at end of EXT-X-PROGRAM-DATE-TIME
					//write EXT-X-DATERANGE end
					PgDataEnd = true
					FgContents = stream.ContentsEnd
				}
				isYolo := ffmpeg.GetYoloDetectorID()
				if len(contents) > 0 && isYolo < 0 {
					glog.Infof("Get Dnn filtering contents at pid %v :%v\n", pid, contents)
					if err := hlsStream.AddHLSSegment(&stream.HLSSegment{SeqNo: seg.SeqNo, Name: sName, Data: warningbuff,
						Duration: 2, PgDataTime: PgDataTime, PgDataEnd: PgDataEnd, FgContents: FgContents, ObjectData: contents, IsYolo: isYolo}); err != nil {
						glog.Errorf("Error writing transcoded seg: %v", err)
					}
				} else {
					if err := hlsStream.AddHLSSegment(&stream.HLSSegment{SeqNo: seg.SeqNo, Name: sName, Data: tData[i],
						Duration: 2, PgDataTime: PgDataTime, PgDataEnd: PgDataEnd, FgContents: FgContents, ObjectData: contents, IsYolo: isYolo}); err != nil {
						glog.Errorf("Error writing transcoded seg: %v", err)
					}
				}
			}
		}
		precontents = contents
	}

	return subscriber, nil
}

func CreateVideoProfile(profiles []ffmpeg.VideoProfile) {
	// interval := 1.5
	// metamode := 2

	// flag.Set("logtostderr", "true")

	dir, err := os.Getwd()
	if err != nil {
		glog.Infof("Error getting work directory: %v", err)
	}
	glog.Infof("Settig working directory %v", fmt.Sprintf("%v/.tmp", dir))
	// lpms := core.New(&core.LPMSOpts{WorkDir: fmt.Sprintf("%v/.tmp", dir)})

	//Streams needed for transcoding:
	// var rtmpStrm stream.RTMPVideoStream
	// var hlsStrm stream.HLSVideoStream
	// var manifest stream.HLSVideoManifest
	// var cancelSeg context.CancelFunc

	//loading dnnmodule only once
	//ffmpeg.InitDnnEngine(ffmpeg.PDnnDetector)
	//Register Dnn filter into Transcode Engine
	cengginflag := true
	ffmpeg.SetCengineFlag(cengginflag)
	ffmpeg.SetAvailableGpuNum(1)
	ffmpeg.SetParallelGpuNum(1)

	for _, ft := range profiles {
		glog.Infof("Registry DnnEngine: name: %v metadamode: %v", ft.Name, ft.Detector.MetaMode)
		ffmpeg.RegistryDnnEngine(ft)
	}

}
