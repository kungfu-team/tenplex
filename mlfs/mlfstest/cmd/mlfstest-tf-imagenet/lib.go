package main

import (
	"strconv"

	"github.com/lgarithm/proc"
)

type (
	P = proc.P
)

var (
	seq    = proc.Seq
	pc     = proc.PC
	ignore = proc.Ignore
	echo   = proc.Echo
	try    = proc.Try

	str = strconv.Itoa
)

func dockerExec(cmd string, args ...string) P {
	ss := []string{
		`exec`, `-t`, name,
		cmd,
	}
	ss = append(ss, args...)
	return pc(`docker`, ss...)
}

func dockerCp(a, b string) P {
	return pc(`docker`, `cp`, a, name+`:`+b)
}

func If(ok bool, p P) P {
	if ok {
		return p
	}
	return seq()
}
