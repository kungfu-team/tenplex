package main

import (
	"context"

	"github.com/lgarithm/proc"
)

var (
	seq   = proc.Seq
	echo  = proc.Echo
	Stdio = proc.Stdio
	shell = proc.Shell
)

type (
	P    = proc.P
	Proc = proc.Proc
)

func Scp(filename, host string, path string) P {
	target := host + ":" + path
	p0 := Proc{
		Prog: `scp`,
		Args: []string{filename, target},
	}
	p1 := echo("done scp: " + filename + " to " + target)
	return seq(shell(p0.CmdCtx(context.TODO())), p1)
}
