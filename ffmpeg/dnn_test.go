package ffmpeg

import (
	"os"
	"strings"
	"testing"
)

func TestDnn_LoadModel(t *testing.T) {
	//MODEL_PARAM=tmodel.pb,input_1,reshape_3/Reshape
	param := os.Getenv("MODEL_PARAM")
	if param == "" {
		t.Skip("Skipping model loading test; no MODEL_PARAM set")
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

//go test -bench .
func BenchmarkDnn_Executetime(b *testing.B) {

	dnncfg := PDnnDetector

	dnnfilter := NewDnnFilter()
	dnnfilter.dnncfg = dnncfg
	if dnnfilter.InitDnnFilter(dnncfg) != true {
		b.Errorf("Can not load model file %v", dnncfg.Detector.ModelPath)
	}

	infname := "../data/bunny2.mp4"

	for i := 0; i < b.N; i++ {
		dnnfilter.ExecuteDnnFilter(infname, Nvidia)
	}

	dnnfilter.StopDnnFilter()

}
