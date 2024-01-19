package experiments

import (
	"fmt"
	"log"
	"sync"
)

type Setup struct {
	Prefix   string
	NWorkers int
	Group    string

	IPs    []string
	PubIPs []string
}

func NewSetup(p string, n int, g string) *Setup {
	s := &Setup{
		Prefix:   p,
		NWorkers: n,
		Group:    g,
	}
	s.GetIPs()
	return s
}

func (s Setup) Names() []string {
	var names []string
	for i := 0; i < s.NWorkers; i++ {
		names = append(names, fmt.Sprintf("%s%02d", s.Prefix, i+1))
	}
	return names
}

func (s *Setup) GetIPs() {
	names := s.Names()
	ips := make([]string, len(names))
	pubIPs := make([]string, len(names))
	{
		var wg sync.WaitGroup
		for i := range names {
			wg.Add(1)
			go func(i int) {
				pubIPs[i] = getPubIP(names[i], s.Group)
				wg.Done()
			}(i)
			wg.Add(1)
			go func(i int) {
				ips[i] = getIP(names[i], s.Group)
				wg.Done()
			}(i)
		}
		wg.Wait()
	}
	//
	for i, name := range names {
		if len(pubIPs[i]) == 0 {
			pubIPs[i] = ips[i]
		}
		log.Printf("public IP of %s: %s", name, pubIPs[i])
	}
	//
	s.IPs = ips
	s.PubIPs = pubIPs
}
