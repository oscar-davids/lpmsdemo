package ffmpeg

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

//go test -run Dnn
func TestDnn_LoadModel(t *testing.T) {
	//MODEL_PARAM=tmodel.pb,input_1,reshape_3/Reshape
	param := os.Getenv("MODEL_PARAM")
	if param == "" {
		t.Skip("Skipping model loading test; no MODEL_PARAM set")
		//t.Error("no MODEL_PARAM set")
	}
	str2dnnfilter := func(inp string) VideoProfile {
		dnnfilter := VideoProfile{}
		strs := strings.Split(inp, ",")
		if len(strs) != 3 {
			return dnnfilter
		}
		dnnfilter.Detector.ModelPath = strs[0]
		dnnfilter.Detector.Input = strs[1]
		dnnfilter.Detector.Output = strs[2]
		return dnnfilter
	}

	_, dir := setupTest(t)
	defer os.RemoveAll(dir)

	dnncfg := str2dnnfilter(param)

	if len(dnncfg.Detector.ModelPath) <= 0 || len(dnncfg.Detector.Input) <= 0 || len(dnncfg.Detector.Output) <= 0 {
		t.Errorf("invalid MODEL_PARAM set %v", param)
	}

	dnnfilter := NewDnnFilter()
	dnnfilter.dnncfg = dnncfg
	if dnnfilter.InitDnnFilter(dnncfg) != true {
		t.Errorf("Can not load model file %v", dnncfg.Detector.ModelPath)
	}
	dnnfilter.StopDnnFilter()
}

func TestDnn_GetVideoInfo(t *testing.T) {

	fname := "../data/bunny2.mp4"

	tvinfo := NewDnnVinfo()
	vinfo := tvinfo.GetVideoInfo(fname)
	if len(vinfo) == 0 {
		t.Errorf("Could not get video information %v", fname)
	} else {
		t.Logf("%v : %v", fname, vinfo)
		fmt.Println(vinfo)
	}

	tvinfo.DeleteDnnVinfo()
}

//go test -run Dnn -bench Dnn
func BenchmarkDnn_Loadtime(b *testing.B) {

	dnncfg := PDnnDetector

	dnnfilter := NewDnnFilter()
	dnnfilter.dnncfg = dnncfg
	if dnnfilter.InitDnnFilter(dnncfg) != true {
		b.Errorf("Can not load model file %v", dnncfg.Detector.ModelPath)
	}

	dnnfilter.StopDnnFilter()

}

//go test -run DnnSW -bench DnnSW
func BenchmarkDnnSW_Transcodingtime(b *testing.B) {

	dnncfg := PDnnDetector

	dnnfilter := NewDnnFilter()
	dnnfilter.dnncfg = dnncfg
	if dnnfilter.InitDnnFilter(dnncfg) != true {
		b.Errorf("Can not load model file %v", dnncfg.Detector.ModelPath)
	}

	var err error
	fname := "../data/bunny2.mp4"
	oname := "outbunny2.ts"
	prof := P720p25fps16x9
	for i := 0; i < b.N; i++ {
		// hw dec, hw enc
		err = Transcode2(&TranscodeOptionsIn{
			Fname: fname,
			Accel: Software,
		}, []TranscodeOptions{
			{
				Oname:   oname,
				Profile: prof,
				Accel:   Software,
			},
		})
		if err != nil {
			b.Error(err)
		}
		os.Remove(oname)
	}

	dnnfilter.StopDnnFilter()

}
func BenchmarkDnnSW_FilterExecutetime(b *testing.B) {

	dnncfg := PDnnDetector

	dnnfilter := NewDnnFilter()
	dnnfilter.dnncfg = dnncfg
	if dnnfilter.InitDnnFilter(dnncfg) != true {
		b.Errorf("Can not load model file %v", dnncfg.Detector.ModelPath)
	}

	fname := "../data/bunny2.mp4"

	for i := 0; i < b.N; i++ {
		dnnfilter.ExecuteDnnFilter(fname, Software)
	}

	dnnfilter.StopDnnFilter()

}
func BenchmarkDnnSW_AllExecutetime(b *testing.B) {

	dnncfg := PDnnDetector

	dnnfilter := NewDnnFilter()
	dnnfilter.dnncfg = dnncfg
	if dnnfilter.InitDnnFilter(dnncfg) != true {
		b.Errorf("Can not load model file %v", dnncfg.Detector.ModelPath)
	}
	var err error
	fname := "../data/bunny2.mp4"
	oname := "outbunny2.ts"
	prof := P720p25fps16x9
	for i := 0; i < b.N; i++ {
		// dnn filter
		dnnfilter.ExecuteDnnFilter(fname, Software)
		// hw dec, hw enc
		err = Transcode2(&TranscodeOptionsIn{
			Fname: fname,
			Accel: Software,
		}, []TranscodeOptions{
			{
				Oname:   oname,
				Profile: prof,
				Accel:   Software,
			},
		})
		if err != nil {
			b.Error(err)
		}

		os.Remove(oname)
	}

	dnnfilter.StopDnnFilter()
}
func BenchmarkDnnSW_AllDigitalExecutetime(b *testing.B) {

	totalreport := ""
	//0x no classify only transcoding
	//1x ,2x, 3x, ... 6x
	for s := 0; s <= 64; {
		dnncfg := PDnnDetector
		dnncfg.Detector.SampleRate = uint(s)
		dnncfg.Detector.Interval = 0.0

		dnnfilter := NewDnnFilter()
		dnnfilter.dnncfg = dnncfg
		if dnnfilter.InitDnnFilter(dnncfg) != true {
			b.Errorf("Can not load model file %v", dnncfg.Detector.ModelPath)
		}
		var err error
		fname := "../data/bunny2.mp4"
		oname := "outbunny2.ts"
		prof := P720p25fps16x9
		start := time.Now()
		for i := 0; i < b.N; i++ {
			// dnn filter
			dnnfilter.ExecuteDnnFilter(fname, Nvidia)
			// hw dec, hw enc
			err = Transcode2(&TranscodeOptionsIn{
				Fname: fname,
				Accel: Nvidia,
			}, []TranscodeOptions{
				{
					Oname:   oname,
					Profile: prof,
					Accel:   Nvidia,
				},
			})
			if err != nil {
				b.Error(err)
			}

			os.Remove(oname)
		}
		elapsed := time.Since(start)
		elasec := elapsed.Seconds() / float64(b.N)
		dnnfilter.StopDnnFilter()

		tvinfo := NewDnnVinfo()
		vinfo := tvinfo.GetVideoInfo(fname)
		tvinfo.DeleteDnnVinfo()
		//run 1x = classify(1) and transcoding(1280x720 30fps) at realtime video len(s): xxxx s
		evalreport := fmt.Sprintf("Run %vx = classify(1) and transcoding(%v %v) at video(%v) : %v sec\n", dnncfg.Detector.SampleRate,
			prof.Resolution, prof.Framerate, vinfo, elasec)

		if s == 0 {
			s = 1
			evalreport = strings.Replace(evalreport, "classify(1)", "no classify", 1)
		} else {
			s *= 2
		}
		totalreport += evalreport

		fmt.Println(evalreport)

	}

	fmt.Println(totalreport)
}

//go test -run DnnHW -bench DnnHW
func BenchmarkDnnHW_Transcodingtime(b *testing.B) {

	dnncfg := PDnnDetector

	dnnfilter := NewDnnFilter()
	dnnfilter.dnncfg = dnncfg
	if dnnfilter.InitDnnFilter(dnncfg) != true {
		b.Errorf("Can not load model file %v", dnncfg.Detector.ModelPath)
	}

	var err error
	fname := "../data/bunny2.mp4"
	oname := "outbunny2.ts"
	prof := P720p25fps16x9
	for i := 0; i < b.N; i++ {
		// hw dec, hw enc
		err = Transcode2(&TranscodeOptionsIn{
			Fname: fname,
			Accel: Nvidia,
		}, []TranscodeOptions{
			{
				Oname:   oname,
				Profile: prof,
				Accel:   Nvidia,
			},
		})
		if err != nil {
			b.Error(err)
		}
		os.Remove(oname)
	}

	dnnfilter.StopDnnFilter()

}
func BenchmarkDnnHW_FilterExecutetime(b *testing.B) {

	dnncfg := PDnnDetector

	dnnfilter := NewDnnFilter()
	dnnfilter.dnncfg = dnncfg
	if dnnfilter.InitDnnFilter(dnncfg) != true {
		b.Errorf("Can not load model file %v", dnncfg.Detector.ModelPath)
	}

	fname := "../data/bunny2.mp4"

	for i := 0; i < b.N; i++ {
		dnnfilter.ExecuteDnnFilter(fname, Nvidia)
	}

	dnnfilter.StopDnnFilter()

}
func BenchmarkDnnHW_AllExecutetime(b *testing.B) {

	dnncfg := PDnnDetector

	dnnfilter := NewDnnFilter()
	dnnfilter.dnncfg = dnncfg
	if dnnfilter.InitDnnFilter(dnncfg) != true {
		b.Errorf("Can not load model file %v", dnncfg.Detector.ModelPath)
	}
	var err error
	fname := "../data/bunny2.mp4"
	oname := "outbunny2.ts"
	prof := P720p25fps16x9
	for i := 0; i < b.N; i++ {
		// dnn filter
		dnnfilter.ExecuteDnnFilter(fname, Nvidia)
		// hw dec, hw enc
		err = Transcode2(&TranscodeOptionsIn{
			Fname: fname,
			Accel: Nvidia,
		}, []TranscodeOptions{
			{
				Oname:   oname,
				Profile: prof,
				Accel:   Nvidia,
			},
		})
		if err != nil {
			b.Error(err)
		}

		os.Remove(oname)
	}

	dnnfilter.StopDnnFilter()
}
func BenchmarkDnnHW_AllDigitalExecutetime(b *testing.B) {

	totalreport := ""
	//0x no classify only transcoding
	//1x ,2x, 3x, ... 6x
	for s := 0; s <= 64; {
		dnncfg := PDnnDetector
		dnncfg.Detector.SampleRate = uint(s)
		dnncfg.Detector.Interval = 0.0

		dnnfilter := NewDnnFilter()
		dnnfilter.dnncfg = dnncfg
		if dnnfilter.InitDnnFilter(dnncfg) != true {
			b.Errorf("Can not load model file %v", dnncfg.Detector.ModelPath)
		}
		var err error
		fname := "../data/bunny2.mp4"
		oname := "outbunny2.ts"
		prof := P720p25fps16x9
		start := time.Now()
		for i := 0; i < b.N; i++ {
			// dnn filter
			dnnfilter.ExecuteDnnFilter(fname, Nvidia)
			// hw dec, hw enc
			err = Transcode2(&TranscodeOptionsIn{
				Fname: fname,
				Accel: Nvidia,
			}, []TranscodeOptions{
				{
					Oname:   oname,
					Profile: prof,
					Accel:   Nvidia,
				},
			})
			if err != nil {
				b.Error(err)
			}

			os.Remove(oname)
		}
		elapsed := time.Since(start)
		elasec := elapsed.Seconds() / float64(b.N)
		dnnfilter.StopDnnFilter()

		tvinfo := NewDnnVinfo()
		vinfo := tvinfo.GetVideoInfo(fname)
		tvinfo.DeleteDnnVinfo()
		//run 1x = classify(1) and transcoding(1280x720 30fps) at realtime video len(s): xxxx s
		evalreport := fmt.Sprintf("Run %vx = classify(1) and transcoding(%v %v) at video(%v) : %v sec\n", dnncfg.Detector.SampleRate,
			prof.Resolution, prof.Framerate, vinfo, elasec)

		if s == 0 {
			s = 1
			evalreport = strings.Replace(evalreport, "classify(1)", "no classify", 1)
		} else {
			s *= 2
		}
		totalreport += evalreport

		fmt.Println(evalreport)

	}

	fmt.Println(totalreport)
}
