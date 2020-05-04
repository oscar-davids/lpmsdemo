package ffmpeg

import (
	"os"
	"strings"
	"testing"
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
