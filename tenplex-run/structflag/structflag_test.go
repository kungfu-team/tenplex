package structflag_test

import (
	"flag"
	"strings"
	"testing"

	"github.com/kungfu-team/tenplex/tenplex-run/structflag"
)

type App struct {
	Name string `flag:"name"`
	X    int    `flag:"x"`
}

func Test_1(t *testing.T) {
	var a App
	f := flag.NewFlagSet(`cmd`, flag.ExitOnError)
	structflag.RegisterFlags(&a, f)
}

func Test_2(t *testing.T) {
	a := App{
		Name: `abc`,
		X:    2,
	}
	args := structflag.ToArgs(&a)
	want := `-name abc -x 2`
	if got := strings.Join(args, " "); got != want {
		t.Errorf("%q != %q", got, want)
	}
}
