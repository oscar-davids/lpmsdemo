package stream

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/oscar-davids/lpmsdemo/m3u8"
)

const DefaultHLSStreamCap = uint(500)
const DefaultHLSStreamWin = uint(3)

// const DefaultMediaWinLen = uint(5)
const DefaultSegWaitTime = time.Second * 10
const SegWaitInterval = time.Second

var ErrAddHLSSegment = errors.New("ErrAddHLSSegment")

//BasicHLSVideoStream is a basic implementation of HLSVideoStream
type BasicHLSVideoStream struct {
	plCache    *m3u8.MediaPlaylist //StrmID -> MediaPlaylist
	segMap     map[string]*HLSSegment
	segNames   []string
	lock       sync.Locker
	strmID     string
	subscriber func(*HLSSegment, bool)
	winSize    uint
}

func NewBasicHLSVideoStream(strmID string, wSize uint) *BasicHLSVideoStream {
	pl, err := m3u8.NewMediaPlaylist(wSize, DefaultHLSStreamCap)
	if err != nil {
		return nil
	}

	return &BasicHLSVideoStream{
		plCache:  pl,
		segMap:   make(map[string]*HLSSegment),
		segNames: make([]string, 0),
		lock:     &sync.Mutex{},
		strmID:   strmID,
		winSize:  wSize,
	}
}

//SetSubscriber sets the callback function that will be called when a new hls segment is inserted
func (s *BasicHLSVideoStream) SetSubscriber(f func(seg *HLSSegment, eof bool)) {
	s.subscriber = f
}

//GetStreamID returns the streamID
func (s *BasicHLSVideoStream) GetStreamID() string { return s.strmID }

func (s *BasicHLSVideoStream) AppData() AppData { return nil }

//GetStreamFormat always returns HLS
func (s *BasicHLSVideoStream) GetStreamFormat() VideoFormat { return HLS }

//GetStreamPlaylist returns the media playlist represented by the streamID
func (s *BasicHLSVideoStream) GetStreamPlaylist() (*m3u8.MediaPlaylist, error) {
	if s.plCache.Count() < s.winSize {
		return nil, nil
	}

	return s.plCache, nil
}

//GetHLSSegment gets the HLS segment.  It blocks until something is found, or timeout happens.
func (s *BasicHLSVideoStream) GetHLSSegment(segName string) (*HLSSegment, error) {
	seg, ok := s.segMap[segName]
	if !ok {
		return nil, ErrNotFound
	}
	return seg, nil
}

//AddHLSSegment adds the hls segment to the right stream
func (s *BasicHLSVideoStream) AddHLSSegment(seg *HLSSegment) error {
	if _, ok := s.segMap[seg.Name]; ok {
		return nil //Already have the seg.
	}
	// glog.V(common.VERBOSE).Infof("Adding segment: %v", seg.Name)

	s.lock.Lock()
	defer s.lock.Unlock()

	now := time.Now()
	//if seg.PgDataEnd == true {
	//	//s.plCache.SetDiscontinuity()
	//	s.plCache.SetProgramDateTime(now)
	//}
	if seg.FgContents == ContentsStart || seg.FgContents == ContentsEnd {
		s.plCache.SetProgramDateTime(now)
	}
	SCTE35Intag := ""
	SCTE35Commandtag := ""
	SCTE35Outtag := ""
	switch seg.FgContents {
	case ContentsStart:
		SCTE35Outtag = "test"
	case ContentsContinue:
		SCTE35Commandtag = "test"
	case ContentsEnd:
		SCTE35Intag = "test"
	case ContentsNone:
	}

	nanosec := int64(seg.Duration * 1000000000.0)
	nowplus := now.Add(time.Duration(nanosec))
	nowplusend := now.Add(time.Duration(nanosec + nanosec))
	if seg.FgContents == ContentsStart || seg.FgContents == ContentsEnd || seg.IsYolo >= 0 {
		DateRange := &m3u8.DateRange{
			ID:               "2020",
			StartDate:        nowplus,
			Duration:         seg.Duration * 2,
			PlannedDuration:  seg.Duration,
			Class:            "class",
			EndDate:          nowplusend,
			ClientAttributes: m3u8.ClientAttributes{"X-AD-URL": "http://ad.com/acme", "detectobjects": seg.ObjectData},
			SCTE35In:         SCTE35Intag,
			SCTE35Out:        SCTE35Outtag,
			SCTE35Command:    SCTE35Commandtag,
		}
		s.plCache.SetDateRange(DateRange)
	}
	//Add segment to media playlist and buffer
	s.plCache.AppendSegment(&m3u8.MediaSegment{SeqId: seg.SeqNo, Duration: seg.Duration, URI: seg.Name})
	s.segNames = append(s.segNames, seg.Name)
	s.segMap[seg.Name] = seg
	if s.plCache.Count() > s.winSize {
		s.plCache.Remove()
		toRemove := s.segNames[0]
		delete(s.segMap, toRemove)
		s.segNames = s.segNames[1:]
	}

	//Call subscriber
	//if s.subscriber != nil {
	//	s.subscriber(seg, false)
	//}

	return nil
}
func (s *BasicHLSVideoStream) TrigerTranscode(seg *HLSSegment) error {
	//Call subscriber
	if s.subscriber != nil {
		s.subscriber(seg, false)
	}
	return nil
}

func (s *BasicHLSVideoStream) End() {
	if s.subscriber != nil {
		s.subscriber(nil, true)
	}
}

func (s BasicHLSVideoStream) String() string {
	return fmt.Sprintf("StreamID: %v, Type: %v, len: %v", s.GetStreamID(), s.GetStreamFormat(), len(s.segMap))
}
