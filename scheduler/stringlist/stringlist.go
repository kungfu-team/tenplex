package stringlist

import (
	"flag"
	"strings"
)

type Value []string

func (v *Value) Set(args string) error {
	for _, t := range strings.Split(args, ",") {
		*v = append(*v, strings.TrimSpace(t))
	}
	return nil
}

func (v *Value) String() string { return strings.Join(*v, ",") }

func Flag(name string, v Value, usage string) *Value {
	r := make(Value, len(v))
	copy(r, v)
	flag.Var(&r, name, usage)
	return &r
}
