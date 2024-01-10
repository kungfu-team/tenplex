package job

import "fmt"

var (
	// komodo01 = `komodo01.doc.res.ic.ac.uk`
	komodo01 = `10.10.10.1`
	// komodo02 = `komodo02.doc.res.ic.ac.uk`
	komodo02 = `10.10.10.2`
	// komodo03 = `komodo03.doc.res.ic.ac.uk`
	komodo03 = `10.10.10.3`
	// komodo04 = `komodo04.doc.res.ic.ac.uk`
	komodo04 = `10.10.10.4`

	dockerIPs = genDockerIPRange(32)
)

// generate a private IP range for docker swarm
func genDockerIPRange(n int) []string {
	var ips []string
	for i := 0; i < n; i++ {
		// TODO: extract subnet range from JSON
		// ip := fmt.Sprintf("10.10.10.%d", 140+i)
		ip := fmt.Sprintf("trainer-%02d", i)
		ips = append(ips, ip)
	}
	return ips
}
