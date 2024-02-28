package listflag

import (
	"flag"
	"strconv"
	"strings"
)

type Strings []string

func (v *Strings) Set(args string) error {
	*v = nil
	for _, t := range strings.Split(args, ",") {
		*v = append(*v, strings.TrimSpace(t))
	}
	return nil
}

func (v *Strings) String() string { return strings.Join(*v, ",") }

func String(name string, v Strings, usage string) *Strings {
	r := make(Strings, len(v))
	copy(r, v)
	flag.Var(&r, name, usage)
	return &r
}

type Ints []int

func (v *Ints) Set(args string) error {
	*v = nil
	for _, t := range strings.Split(args, ",") {
		s := strings.TrimSpace(t)
		n, err := strconv.Atoi(s)
		if err != nil {
			return err
		}
		*v = append(*v, n)
	}
	return nil
}

func (v *Ints) String() string {
	var ss []string
	for _, n := range *v {
		ss = append(ss, strconv.Itoa(n))
	}
	return strings.Join(ss, ",")
}

func Int(name string, v Ints, usage string) *Ints {
	r := make(Ints, len(v))
	copy(r, v)
	flag.Var(&r, name, usage)
	return &r
}
