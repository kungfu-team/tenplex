package cancelgroup

import (
	"context"

	"github.com/lgarithm/proc"
)

type (
	P = proc.P
)

func CancelGroup(ps []P, defaultErr error, cancel context.CancelFunc) P {
	var qs []P
	for _, p := range ps {
		var err error = defaultErr
		qs = append(qs,
			proc.Seq(
				proc.Ignore(
					proc.Seq(
						p,
						proc.FnOk(func() { err = nil }),
					),
				),
				proc.Fn(func() error {
					if err != nil {
						cancel()
					}
					return err
				}),
			))
	}
	return proc.Par(qs...)
}
