package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/jinzhu/gorm"
	_ "github.com/jinzhu/gorm/dialects/sqlite"
	// detection "github.com/woody0105/lpmsdemo/detection"
)

type User struct {
	gorm.Model
	Name  string
	Email string
}

type UserRequest struct {
	Name  string `json: "name"`
	Email string `json: "email"`
}

var startstreamfunc = detection.startStream

// type StreamRequest struct {
// 	Name     string                `json: "name"`
// 	Profiles []ffmpeg.VideoProfile `json: "profiles"`
// }

type ProfileRequest struct {
	Name        string
	Bitrate     string
	Fps         uint
	Resolution  string
	AspectRatio string
	Detector    string
}

type StreamRequest struct {
	Name     string
	Profiles []ProfileRequest
}

// type StreamRequest struct{
// 	Name string `json: "name"`
// }

func allUsers(w http.ResponseWriter, r *http.Request) {
	db, err := gorm.Open("sqlite3", "test.db")
	if err != nil {
		panic("failed to connect database")
	}
	defer db.Close()

	var users []User
	db.Find(&users)
	fmt.Println("{}", users)

	json.NewEncoder(w).Encode(users)
}

func newUser(w http.ResponseWriter, r *http.Request) {
	fmt.Println("New User Endpoint Hit")
	w.Header().Set("Content-Type", "application/json")
	db, err := gorm.Open("sqlite3", "test.db")
	if err != nil {
		panic("failed to connect database")
	}
	defer db.Close()
	var userRequest UserRequest
	json.NewDecoder(r.Body).Decode(&userRequest)
	fmt.Println(userRequest.Name)
	fmt.Println(userRequest.Email)

	db.Create(&User{Name: userRequest.Name, Email: userRequest.Email})
	fmt.Fprintf(w, "New User Successfully Created")
}

func newStream(w http.ResponseWriter, r *http.Request) {
	fmt.Println("New Stream Endpoint Hit")
	w.Header().Set("Content-Type", "application/json")
	db, err := gorm.Open("sqlite3", "test.db")
	if err != nil {
		panic("failed to connect database")
	}
	defer db.Close()
	var streamRequest StreamRequest
	reqBody, _ := ioutil.ReadAll(r.Body)
	json.Unmarshal(reqBody, &streamRequest)

	fmt.Println(streamRequest.Name)
	var profiles []ProfileRequest
	profiles = streamRequest.Profiles
	fmt.Println(profiles)
	fmt.Println(profiles[0].Name, profiles[0].Bitrate, profiles[0].Detector)
	detection.startStream()

	fmt.Fprintf(w, "New Stream Successfully Created")
}

func handleRequests() {
	myRouter := mux.NewRouter().StrictSlash(true)
	myRouter.HandleFunc("/users", allUsers).Methods("GET")
	myRouter.HandleFunc("/user", newUser).Methods("POST")
	myRouter.HandleFunc("/stream", newStream).Methods("POST")
	log.Fatal(http.ListenAndServe(":8081", myRouter))
}

func initialMigration() {
	db, err := gorm.Open("sqlite3", "test.db")
	if err != nil {
		fmt.Println(err.Error())
		panic("failed to connect database")
	}
	defer db.Close()

	// Migrate the schema
	db.AutoMigrate(&User{})
}

func main() {
	fmt.Println("Go ORM Tutorial")

	initialMigration()
	// Handle Subsequent requests
	handleRequests()
}
