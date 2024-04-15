package structflag_test

import (
	"flag"
	"strings"
	"testing"

	"github.com/kungfu-team/tenplex/tenplex-run/structflag"
)

type Base struct {
	Y int `flag:"y"`
}

type App struct {
	Base
	Name string `flag:"name"`
	X    int    `flag:"x"`
	OK   bool   `flag:"ok"`
}

func Test_1(t *testing.T) {
	var a App
	f := flag.NewFlagSet(`cmd`, flag.ExitOnError)
	structflag.RegisterFlags(&a, f) // won't register Base
	// structflag.RegisterFlags(&a.Base, f)
}

func Test_2(t *testing.T) {
	a := App{
		Name: `abc`,
		X:    2,
		OK:   true,
	}
	args := structflag.ToGoArgs(&a)
	want := `-name abc -x 2 -ok`
	if got := strings.Join(args, " "); got != want {
		t.Errorf("%q != %q", got, want)
	}
}
