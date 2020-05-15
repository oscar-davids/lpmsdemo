/*
The Example Media Server.  It takes an RTMP stream, segments it into a HLS stream, and transcodes it so it's available for Adaptive Bitrate Streaming.
*/
package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"
	"io/ioutil"

	"github.com/oscar-davids/lpmsdemo/transcoder"

	"github.com/golang/glog"
	"github.com/oscar-davids/lpmsdemo/core"
	"github.com/oscar-davids/lpmsdemo/ffmpeg"
	"github.com/oscar-davids/lpmsdemo/segmenter"
	"github.com/oscar-davids/lpmsdemo/stream"
	"github.com/oscar-davids/lpmsdemo/m3u8"
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

//main -classid=0 -interval=1.5 -dnnfilter=PDnnDetector,PDnnVioFilter,PDnnOtherFilter -metamode=1 -gpucount=2 -parallel=2
func main() {

	strfilters := flag.String("dnnfilter", "PDnnDetector", "dnn filters for classification")
	flagClass := flag.Int("classid", 0, "class id for classification")
	interval := flag.Float64("interval", 1.0, "time interval(unit second) for classification")
	metaMode := flag.Int("metamode", 0, "metadata store mode(default subtitle 0) about output pmegts file")
	gpucount := flag.Int("gpucount", 1, "avaible gpu count for clasiifier and transcoding")
	parallel := flag.Int("parallel", 2, "parallel processing count for clasiifier")
	flag.Parse()
	if flag.Parsed() == false || *interval <= float64(0.0) {
		panic("Usage sample: appname -classid=0 -interval=1.5 -dnnfilter=PDnnDetector -metamode=0 -gpucount=2 -parallel=2")
	}
	for i, s := range os.Args {
		if i == 0 {
			continue
		}
		if strings.Index(s, "-classid=") < 0 && strings.Index(s, "-interval=") < 0 && strings.Index(s, "-parallel=") < 0 &&
			strings.Index(s, "-dnnfilter=") < 0 && strings.Index(s, "-metamode=") < 0 && strings.Index(s, "-gpucount=") < 0 {
			panic("Usage sample: appname -classid=0 -interval=1.5 -dnnfilter=PDnnDetector -metamode=0 -gpucount=2 -parallel=2")
		}
	}
	//check dnnfilter
	str2filters := func(inp string) []ffmpeg.VideoProfile {
		filters := []ffmpeg.VideoProfile{}
		strs := strings.Split(inp, ",")
		for _, k := range strs {
			if strings.Index(k, "PDnn") < 0 {
				continue
			}
			p, ok := ffmpeg.VideoProfileLookup[k]
			if !ok {
				panic(fmt.Sprintf("Invalid DnnFilter %s. Valid DnnFilters are:\n%s", k, validDnnfilters()))
			}
			filters = append(filters, p)
		}
		return filters
	}

	dnnfilters := str2filters(*strfilters)

	flag.Set("logtostderr", "true")
	flag.Parse()

	dir, err := os.Getwd()
	if err != nil {
		glog.Infof("Error getting work directory: %v", err)
	}
	glog.Infof("Settig working directory %v", fmt.Sprintf("%v/.tmp", dir))
	lpms := core.New(&core.LPMSOpts{WorkDir: fmt.Sprintf("%v/.tmp", dir)})

	//Streams needed for transcoding:
	var rtmpStrm stream.RTMPVideoStream
	var hlsStrm stream.HLSVideoStream
	var manifest stream.HLSVideoManifest
	var cancelSeg context.CancelFunc
	//loading dnnmodule only once
	//ffmpeg.InitDnnEngine(ffmpeg.PDnnDetector)
	//Register Dnn filter into Transcode Engine
	ffmpeg.SetAvailableGpuNum(*gpucount)
	ffmpeg.SetParallelGpuNum(*parallel)

	for i, ft := range dnnfilters {
		if i == 0 {
			ft.Detector.ClassID = *flagClass
		}
		ft.Detector.Interval = float32(*interval)
		ft.Detector.MetaMode = *metaMode

		glog.Infof("Registry DnnEngine: name: %v metadamode: %v", ft.Name, ft.Detector.MetaMode)
		ffmpeg.RegistryDnnEngine(ft)
	}

	lpms.HandleRTMPPublish(
		//makeStreamID (give the stream an ID)
		func(url *url.URL) stream.AppData {
			s := exampleStream(randString(10))
			return &s
		},

		//gotStream
		func(url *url.URL, rs stream.RTMPVideoStream) (err error) {
			//Store the stream
			reqName := getRTMPRequestName(url)
			glog.Infof("Got RTMP stream: %v %v %v", url.Path, reqName, rs.GetStreamID())
			rtmpStrm = rs

			// //Segment the video into HLS (If we need multiple outlets for the HLS stream, we'd need to create a buffer.  But here we only have one outlet for the transcoder)
			hlsStrm = stream.NewBasicHLSVideoStream(randString(10), 8)

			var subscriber func(*stream.HLSSegment, bool)
			subscriber, err = transcode(hlsStrm, *flagClass, *interval)
			if err != nil {
				glog.Errorf("Error transcoding: %v", err)
			}
			hlsStrm.SetSubscriber(subscriber)
			glog.Infof("After set subscriber")

			opt := segmenter.SegmenterOptions{SegLength: 2 * time.Second}
			var ctx context.Context
			ctx, cancelSeg = context.WithCancel(context.Background())

			//Kick off FFMpeg to create segments
			go func() {
				if err := lpms.SegmentRTMPToHLS(ctx, rtmpStrm, hlsStrm, opt); err != nil {
					glog.Errorf("Error segmenting RTMP video stream: %v", err)
				}
			}()
			glog.Infof("HLS StreamID: %v", hlsStrm.GetStreamID())

			mid := reqName + "_classified"
			manifest = stream.NewBasicHLSVideoManifest(mid)
			pl, _ := hlsStrm.GetStreamPlaylist()
			variant := &m3u8.Variant{URI: fmt.Sprintf("%v.m3u8", mid), Chunklist: pl, VariantParams: m3u8.VariantParams{}}
			manifest.AddVideoStream(hlsStrm, variant)

			return nil
		},
		//endStream
		func(url *url.URL, rtmpStrm stream.RTMPVideoStream) error {
			glog.Infof("Ending stream for %v", hlsStrm.GetStreamID())
			//Remove the stream
			streamID := hlsStrm.GetStreamID()
			cancelSeg()
			rtmpStrm = nil
			hlsStrm = nil
			ffmpeg.RemoveParallelID(streamID)
			ffmpeg.RemoveGpuInx(streamID)
			return nil
		})

	lpms.HandleHLSPlay(
		//getMasterPlaylist
		func(url *url.URL) (*m3u8.MasterPlaylist, error) {
			if parseStreamID(url.Path) == "transcoded" && hlsStrm != nil {
				mpl, err := manifest.GetManifest()
				if err != nil {
					glog.Errorf("Error getting master playlist: %v", err)
					return nil, err
				}
				glog.Infof("Master Playlist: %v", mpl.String())
				return mpl, nil
			}
			return nil, nil
		},
		//getMediaPlaylist
		func(url *url.URL) (*m3u8.MediaPlaylist, error) {
			if nil == hlsStrm {
				return nil, fmt.Errorf("No stream available")
			}
			//Wait for the HLSBuffer gets populated, get the playlist from the buffer, and return it.
			start := time.Now()
			for time.Since(start) < HLSWaitTime {
				pl, err := hlsStrm.GetStreamPlaylist()
				if err != nil || pl == nil || pl.Segments == nil || len(pl.Segments) <= 0 || pl.Segments[0] == nil || pl.Segments[0].URI == "" {
					if err == stream.ErrEOF {
						return nil, err
					}

					time.Sleep(2 * time.Second)
					continue
				} else {
					return pl, nil
				}
			}
			return nil, fmt.Errorf("Error getting playlist")
		},
		//getSegment
		func(url *url.URL) ([]byte, error) {
			seg, err := hlsStrm.GetHLSSegment(getHLSSegmentName(url))
			if err != nil {
				glog.Errorf("Error getting segment: %v", err)
				return nil, err
			}
			return seg.Data, nil
		})

	lpms.HandleRTMPPlay(
		//getStream
		func(url *url.URL) (stream.RTMPVideoStream, error) {
			glog.Infof("Got req: %v", url.Path)
			if rtmpStrm != nil {
				strmID := parseStreamID(url.Path)
				if strmID == rtmpStrm.GetStreamID() {
					return rtmpStrm, nil
				}
			}
			return nil, fmt.Errorf("Cannot find stream")
		})

	lpms.Start(context.Background())
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
		glog.Errorf("Can not read temp warning video: %v", err)
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
				if p.Name == "PDnnDetector" {
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
				
				if len(contents) > 0 {
					glog.Infof("Get Dnn filtering contents at pid %v :%v\n", pid, contents)

					if err := hlsStream.AddHLSSegment(&stream.HLSSegment{SeqNo: seg.SeqNo, Name: sName, Data: warningbuff, 
						Duration: 2, PgDataTime: PgDataTime, PgDataEnd: PgDataEnd, FgContents: FgContents}); err != nil {
						glog.Errorf("Error writing transcoded seg: %v", err)
					}
				} else {
					if err := hlsStream.AddHLSSegment(&stream.HLSSegment{SeqNo: seg.SeqNo, Name: sName, Data: tData[i], 
						Duration: 2, PgDataTime: PgDataTime, PgDataEnd: PgDataEnd, FgContents: FgContents}); err != nil {
						glog.Errorf("Error writing transcoded seg: %v", err)
					}
				}
			}
		}
		precontents = contents
	}

	return subscriber, nil
}
