package main

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"os"
	"path"
	"strconv"
	"strings"

	"github.com/golang/glog"

	"github.com/oscar-davids/lpmsdemo/detection"
	"github.com/oscar-davids/lpmsdemo/ffmpeg"
	"github.com/oscar-davids/lpmsdemo/stream"

	"github.com/gorilla/mux"
)

type StreamRequest struct {
	Name     string
	Profiles []ffmpeg.VideoProfile
}

type BroadcasterAddress struct {
	Address string `json:"address"`
}

var transcodeprofiles []ffmpeg.VideoProfile
var broadcaster string
var broadcastercfg string

// RandomIDGenerator generates random hexadecimal string of specified length
// defined as variable for unit tests
var RandomIDGenerator = func(length uint) string {
	x := make([]byte, length, length)
	for i := 0; i < len(x); i++ {
		x[i] = byte(rand.Uint32())
	}
	return hex.EncodeToString(x)
}

func RandName() string {
	return RandomIDGenerator(10)
}

func getBroadcaster(w http.ResponseWriter, r *http.Request) {
	var broadcasteraddress BroadcasterAddress
	if len(broadcaster) > 0 {
		broadcasteraddress = BroadcasterAddress{Address: broadcaster}
	}

	json.NewEncoder(w).Encode(broadcasteraddress)
}

func newStream(w http.ResponseWriter, r *http.Request) {
	fmt.Println("New Stream Endpoint Hit")
	w.Header().Set("Content-Type", "application/json")

	var streamRequest StreamRequest
	reqBody, _ := ioutil.ReadAll(r.Body)
	json.Unmarshal(reqBody, &streamRequest)

	fmt.Println(streamRequest.Name)
	var profiles []ffmpeg.VideoProfile
	profiles = streamRequest.Profiles

	var detectionprofiles []ffmpeg.VideoProfile

	for _, profile := range profiles {
		if strings.Index(profile.Name, "PDnn") < 0 {
			videoprofile := ffmpeg.VideoProfileLookup[profile.Name]
			transcodeprofiles = append(transcodeprofiles, videoprofile)
		} else {
			detectionprofile := ffmpeg.VideoProfileLookup[profile.Name]
			detectionprofiles = append(detectionprofiles, detectionprofile)
		}

	}

	detection.InitDnnEngine(detectionprofiles)

	broadcaster = broadcastercfg

	fmt.Fprintf(w, "New Stream Successfully Created")
}

func handlePush(w http.ResponseWriter, r *http.Request) {
	// we read this unconditionally, mostly for ffmpeg
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		httpErr := fmt.Sprintf(`Error reading http request body: %s`, err.Error())
		glog.Error(httpErr)
		http.Error(w, httpErr, http.StatusInternalServerError)
		return
	}

	r.URL = &url.URL{Scheme: "http", Host: r.Host, Path: r.URL.Path}
	vars := mux.Vars(r)
	stream_id := vars["stream_id"]

	fname := path.Base(r.URL.Path)
	seq, err := strconv.ParseUint(strings.TrimSuffix(fname, ".ts"), 10, 64)
	if err != nil {
		seq = 0
	}

	duration, err := strconv.Atoi(r.Header.Get("Content-Duration"))
	if err != nil {
		duration = 2000
		glog.Info("Missing duration; filling in a default of 2000ms")
	}

	// write ts file to tmp directory
	workDir := ".tmp/"
	file, err := os.Create(workDir + fname)
	if err != nil {
		panic(err)
	}
	file.Write(body)
	file.Close()
	r.Body.Close()

	seg := &stream.HLSSegment{
		Data:     body,
		Name:     fname,
		SeqNo:    seq,
		Duration: float64(duration) / 1000.0,
	}

	// check if stream with transcoding profiles is created
	if len(transcodeprofiles) <= 0 {
		http.Error(w, "No Video Profile Created. Create video profile first and try again.", http.StatusInternalServerError)
		return
	}

	// transcode and detect
	resultdata, contents, err := detection.ProcessSegment(seg, stream_id, transcodeprofiles)
	if err != nil {
		glog.Errorf("Error transcoding: %v", err)
	} else {
		fmt.Println("Contents Detected:", contents)
	}

	os.Remove(workDir + fname)

	boundary := RandName()
	accept := r.Header.Get("Accept")
	if accept == "multipart/mixed" {
		contentType := "multipart/mixed; boundary=" + boundary
		w.Header().Set("Content-Type", contentType)
	}
	w.WriteHeader(http.StatusOK)
	w.(http.Flusher).Flush()
	if accept != "multipart/mixed" {
		return
	}
	mw := multipart.NewWriter(w)

	for i, result := range resultdata {
		mw.SetBoundary(boundary)
		var typ, ext string
		length := len(result)
		ext = ".ts"

		typ = "video/mp2t"

		profile := transcodeprofiles[i].Name
		fname := fmt.Sprintf(`"%s_%d%s"`, profile, seq, ext)
		hdrs := textproto.MIMEHeader{
			"Content-Type":        {typ + "; name=" + fname},
			"Content-Length":      {strconv.Itoa(length)},
			"Content-Disposition": {"attachment; filename=" + fname},
			"Rendition-Name":      {profile},
			"Detection-Result":    {contents},
		}
		fw, err := mw.CreatePart(hdrs)
		if err != nil {
			glog.Error("Could not create multipart part ", err)
		}

		io.Copy(fw, bytes.NewBuffer(result))
	}
	mw.Close()

}

func handleRequests() {
	myRouter := mux.NewRouter().StrictSlash(true)
	// transcoder
	myRouter.HandleFunc("/stream", newStream).Methods("POST")
	myRouter.HandleFunc("/broadcaster", getBroadcaster).Methods("GET")
	// broadcaster
	myRouter.HandleFunc("/live/{stream_id}/{media_number}", handlePush).Methods("POST")
	log.Fatal(http.ListenAndServe(":8080", myRouter))
}

func main() {
	flag.StringVar(&broadcastercfg, "broadcaster", "", "broadcaster address")
	flag.Parse()
	fmt.Println("Starting Rest API Server")
	handleRequests()
}
