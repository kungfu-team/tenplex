package statetransform

import (
	"testing"
)

func TestMapToUnitedRequests(t *testing.T) {
	sourceDim := 256
	targetDim := 512
	sourceMPSize := 4
	targetMPRank := 1
	reqs, err := mapToUnitedRequests(sourceDim, targetDim, sourceMPSize, targetMPRank)
	if err != nil {
		t.Fail()
	}
	t.Logf("requests %v", reqs)
}

func TestMapToSourceRequests(t *testing.T) {
	sourceDim := 256
	targetDim := 512
	sourceMPSize := 4
	targetMPRank := 1
	reqs, err := mapToUnitedRequests(sourceDim, targetDim, sourceMPSize, targetMPRank)
	if err != nil {
		t.Fail()
	}
	reqs = mapToSourceRequests(reqs, sourceDim)
	t.Logf("requests %v", reqs)
}
