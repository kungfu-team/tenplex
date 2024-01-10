package trds

import (
	"fmt"
	"testing"
)

func Test_1(t *testing.T) {
	xs := []int{2, 2, 2, 2, 2}
	ps := groupIntList(xs)
	fmt.Printf("%v\n", ps)
	if len(ps) != 1 {
		t.Fail()
	}
	if ps[0].Second != 5 {
		t.Fail()
	}
}
