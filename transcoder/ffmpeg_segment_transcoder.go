package transcoder

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"

	"github.com/golang/glog"
	"github.com/livepeer/lpms/ffmpeg"
)

//SegmentTranscoder transcodes segments individually.  This is a simple wrapper for calling FFMpeg on the command line.
type FFMpegSegmentTranscoder struct {
	tProfiles  []ffmpeg.VideoProfile
	workDir    string
	streamID   string
	parallelid int
}

func NewFFMpegSegmentTranscoder(ps []ffmpeg.VideoProfile, workd string) *FFMpegSegmentTranscoder {
	return &FFMpegSegmentTranscoder{tProfiles: ps, workDir: workd, parallelid: -1}
}
func (t *FFMpegSegmentTranscoder) SetStreamID(sid string) {
	t.streamID = sid
}
func (t *FFMpegSegmentTranscoder) SetParallelID(pid int) {
	t.parallelid = pid
}
func (t *FFMpegSegmentTranscoder) Transcode(fname string) ([][]byte, error) {
	//Invoke ffmpeg
	err := ffmpeg.Transcode(fname, t.workDir, t.parallelid, t.tProfiles)
	if err != nil {
		glog.Errorf("Error transcoding: %v", err)
		return nil, err
	}

	dout := make([][]byte, len(t.tProfiles), len(t.tProfiles))
	for i, p := range t.tProfiles {
		ofile := path.Join(t.workDir, fmt.Sprintf("out%v%v", i, filepath.Base(fname)))
		//ofile = ".tmp/" + fmt.Sprintf("out%v%v", i, filepath.Base(fname))
		if p.Name == "PDnnDetector" {
			continue
		}
		d, err := ioutil.ReadFile(ofile)
		if err != nil {
			glog.Errorf("Cannot read transcode output: %v", err)
		}
		dout[i] = d
		os.Remove(ofile)
	}

	return dout, nil
}
