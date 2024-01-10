package utils

import (
	"fmt"
	"log"
	"os"
	"time"
)

func LogArgs() {
	for i, a := range os.Args {
		fmt.Printf("[arg] [%d]=%s\n", i, a)
	}
}

func LogEnv() {
	for _, e := range os.Environ() {
		fmt.Printf("[env] %s\n", e)
	}
}

func ShowSize(n int64) string {
	const (
		Ki = 1 << 10
		Mi = 1 << 20
		Gi = 1 << 30
		Ti = 1 << 40
	)
	if n >= Ti {
		return fmt.Sprintf("%.1fTi", float64(n)/float64(Ti))
	} else if n >= Gi {
		return fmt.Sprintf("%.1fGiB", float64(n)/float64(Gi))
	} else if n >= Mi {
		return fmt.Sprintf("%.1fMiB", float64(n)/float64(Mi))
	} else if n >= Ki {
		return fmt.Sprintf("%.1fKiB", float64(n)/float64(Ki))
	}
	return fmt.Sprintf("%d", n)
}

func Percent(p, n int) float64 { return 100.0 * float64(p) / float64(n) }

func LogETA(t0 time.Time, progress, total int) {
	d := time.Since(t0)
	r := Percent(progress, total)
	if progress == 0 {
		log.Printf("%.1f%% took %s, ETA: %s", r, d, `?`)
		return
	}
	remain := time.Duration(float64(d) * float64(total-progress) / float64(progress))
	log.Printf("%.1f%% took %s, ETA: %s", r, d, remain)
}
