
# LPMS - Livepeer media server for scene classification 

This proof of concept is a [lpms](https://github.com/livepeer/lpms) fork project to enable scene classification in LPMS

A Nvidia GPU (pascal or higher) is needed for real-time scene classification and transcoding.

To try this project as a standalone service, follow the instructions below.

You can load any trained classifier and get a propability result during the livestream encoding. 

### Requirements

Project requires libavcodec (ffmpeg) and friends. See `install_ffmpeg.sh` . Running this script will install everything in `~/compiled`. In order to build the project, the dependent libraries will need to be discoverable by pkg-config and golang. If you installed everything with `install_ffmpeg.sh` , then run `export PKG_CONFIG_PATH=~/compiled/lib/pkgconfig:$PKG_CONFIG_PATH` so the deps are picked up.
  
  remark: For rapid quality assurance we offer use of burned in subtitle. To use this ffmpeg should be built with --enable-libass

For the classification need to install tensorflow 1.15 and CUDA 10.0. 

 https://www.tensorflow.org/install/install_c
 
 https://www.tensorflow.org/install/gpu
 
To build the project, you need to install golang.

https://golang.org/doc/install

### Build 

check PKG_CONFIG_PATH environment value.

export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}:$HOME/compiled/lib/pkgconfig"

```
git clone https://github.com/oscar-davids/lpmsdemo.git 

cd lpmsdemo

go build cmd/example/main.go

```

### Testing for classification

If the build successed, you can find the main execute file in the lpmsdemo folder.

For classification needs a trained model file. Please change trained filename(base on tensorflow)  to tmodel.pb  and copy it to the lpmsdemo folder, now you can start the lpms server.

./main or ./main -classid=1 -interval=1.5 -dnnfilter=PDnnDetector,PDnnOtherFilter -metamode=1

    classid : class id for classification
  
    interval: time intervals(unit second) for classification
    
    dnnfilter: dnn model name for classification
    
    metamode: select metadata store type.(0: subtitle(default), 1: ffmpeg metadata, 2: hls timed metadata(reservation)) 
    
remark(dnnfilter): Users can use own trained models after register in [here](https://github.com/oscar-davids/lpmsdemo/blob/b9189028be8454cfc34a7186b38c6bfd642b6ba6/ffmpeg/videoprofile.go#L57.).
The Model trained using the Tensorflow should be a model for classification. In other words, the outputs of the model should be inference values array, not an image buffer.


The test server exposes a few different endpoints:

1. `rtmp://localhost:1935/stream/test` for uploading/viewing RTMP video stream.
2. `http://localhost:7935/stream/test_classified.m3u8` for verification classification the HLS video stream.

  remark: can check deterministic streamname with postfix "_classified" for the classifier output


Do the following steps to view a live stream video:

1. Start LPMS by running `go run cmd/example/main.go`

2. Upload an RTMP video stream to `rtmp://localhost:1935/stream/test`.  We recommend using ffmpeg or [OBS](https://obsproject.com/download).

For ffmpeg on osx, run: `ffmpeg -f avfoundation -framerate 30 -pixel_format uyvy422 -i "0:0" -c:v libx264 -tune zerolatency -b:v 900k -x264-params keyint=60:min-keyint=60 -c:a aac -ac 2 -ar 44100 -f flv rtmp://localhost:1935/stream/test`

For OBS, fill in Settings->Stream->URL to be rtmp://localhost:1935

3. If you have successfully uploaded the stream, you should see something like this in the LPMS output
```
I0324 09:44:14.639405   80673 listener.go:28] RTMP server got upstream
I0324 09:44:14.639429   80673 listener.go:42] Got RTMP Stream: test
```
4. If you have scene classification successfully running, you should see something like this in the output

```
Opening '/home/gpu-user/lpmsdemo/.tmp/fca11fb1bd091e944388.m3u8.tmp' for writing
Got seg: fca11fb1bd091e944388_12.ts fca11fb1bd091e944388

#Engine Probability = 0.800000

Shaper: FriBidi 0.19.7 (SIMPLE) HarfBuzz-ng 1.7.2 (COMPLEX)
Using font provider fontconfig

```


5. Now you have a RTMP video stream running, we can view it from the server.  Simply run `ffplay http://localhost:7935/stream/test_classified.m3u8`, you should see the hls video playback.

### GPU Support

Processing on Nvidia GPUs is supported. To enable this capability, FFmpeg needs
to be built with GPU support. See the
[FFmpeg guidelines](https://trac.ffmpeg.org/wiki/HWAccelIntro#NVENCNVDEC) on
this.

To execute the nvidia tests within the `ffmpeg` directory, run this command:

```
go test -tag=nvidia -run Nvidia

```

To run the tests on a particular GPU, use the GPU_DEVICE environment variable:

```
# Runs on GPU number 3
GPU_DEVICE=3 go test -tag nvidia -run Nvidia
```

Aside from the tests themselves, there is a
[sample program](https://github.com/oscar-davids/lpmsdemo/blob/master/cmd/transcoding/transcoding.go)
that can be used as a reference to the LPMS GPU transcoding API. The sample
program can select GPU or software processing via CLI flags. Run the sample
program via:

```
# software processing
go run cmd/transcoding/transcoding.go transcoder/test.ts P144p30fps16x9,PDnnDetector sw

# nvidia processing, GPU number 2
go run cmd/transcoding/transcoding.go transcoder/test.ts P144p30fps16x9,PDnnDetector nv 0
```

### Benchmark

To execute the test and benchmark within the `ffmpeg` directory, run this command:

```
go test -run=Dnn -bench DnnXX

```
In here if have Nvidia GPU, can run the benchmark and test on GPU, otherwise can run on CPU

```
# Runs on GPU
go test -run=Dnn -bench DnnHW
```

```
# Runs on CPU
go test -run=Dnn -bench DnnSW
```



