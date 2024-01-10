package experiments

import (
	"strings"

	"github.com/kungfu-team/scheduler/azcli"
	"github.com/lgarithm/proc"
)

type (
	At   = proc.UserHost
	P    = proc.P
	Proc = proc.Proc
)

var (
	par    = proc.Par
	out    = proc.Output
	seq    = proc.Seq
	Main   = proc.Main
	psh    = proc.Psh
	at     = proc.At
	echo   = proc.Echo
	lmd    = proc.Lambda
	ignore = proc.Ignore
	urpc   = proc.Urpc
)

func getPubIP(name, group string) string {
	o := string(out(psh(azcli.GetPubIP(name, group))))
	return strings.Trim(o, "\n\"")
}

func getIP(name, group string) string {
	o := string(out(psh(azcli.GetIP(name, group))))
	return strings.Trim(o, "\n\"")
}

func fmap[X any, Y any](f func(X) Y, xs ...X) []Y {
	var ys []Y
	for _, x := range xs {
		ys = append(ys, f(x))
	}
	return ys
}

func parmap[T any](f func(T) P, xs ...T) P { return par(fmap(f, xs...)...) }
