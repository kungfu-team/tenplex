package dbg

import "github.com/lgarithm/proc"

func SSH(p proc.Proc) proc.P {
	target := p.Host
	if len(p.User) > 0 {
		target = p.User + `@` + p.Host
	}
	args := []string{
		`-v`,
		target,
		p.Prog,
	}
	args = append(args, p.Args...)
	return proc.PC(`ssh`, args...)
}
