package job

import (
	"github.com/lgarithm/proc"
)

var (
	Par    = proc.Par // (P, P, ...) -> P
	Seq    = proc.Seq //  (P, ...) -> P
	Term   = proc.Term
	Echo   = proc.Echo
	Shell  = proc.Shell
	Ignore = proc.Ignore
	Run    = proc.Run
	Ssh    = proc.SSH
	// Ssh = dbg.SSH
)

type (
	P    = proc.P
	Proc = proc.Proc
)

func Pmap(f func(string) P, hs ...string) []P {
	var ps []P
	for _, h := range hs {
		ps = append(ps, f(h))
	}
	return ps
}
