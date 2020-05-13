package core

import (
	"context"

	"github.com/oscar-davids/lpmsdemo/segmenter"
	"github.com/oscar-davids/lpmsdemo/stream"
)

//RTMPSegmenter describes an interface for a segmenter
type RTMPSegmenter interface {
	SegmentRTMPToHLS(ctx context.Context, rs stream.RTMPVideoStream, hs stream.HLSVideoStream, segOptions segmenter.SegmenterOptions) error
}
