package listflag_test

import (
	"flag"
	"testing"

	"github.com/kungfu-team/tenplex/tenplex-run/listflag"
)

func isFlagValue(flag.Value) {}

func Test_1(t *testing.T) {
	var x listflag.Strings
	isFlagValue(&x)
}

func Test_2(t *testing.T) {
	var x listflag.Ints
	isFlagValue(&x)
}
