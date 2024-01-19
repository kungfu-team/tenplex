package experiments

import (
	"fmt"
	"strings"

	"github.com/lgarithm/proc"
	"github.com/lgarithm/proc/experimental"
)

const netName = `tenplex`

func SetupSwarm(admin string, pubIPs []string, ips []string) P {
	if len(pubIPs) <= 0 {
		return echo("no workers")
	}
	as := fmap(func(h string) At { return at(admin, h) }, pubIPs...)
	a0 := as[0]
	return seq(
		proc.If(false, seq(
			parmap(addDockerGroup, as...),
			parmap(dockerLogin, as...),
		)),
		cleanupSwarm(admin, pubIPs),
		initSwarmLeader(a0),
		lmd(func() P {
			token := strings.TrimSpace(string(out(getJoinToken(a0))))
			return seq(
				parmap(func(a At) P {
					return seq(
						echo(fmt.Sprintf(`%s using token: %q`, a.Host, token)),
						joinSwarm(a, token, ips[0]),
					)
				}, as[1:]...),
				echo("all workers joined"),
			)
		}),
	)
}

func cleanupSwarm(admin string, pubIPs []string) P {
	as := fmap(func(h string) At { return at(admin, h) }, pubIPs...)
	a0 := as[0]
	return seq(
		parmap(leaveSwarm, as...),
		ignore(urpc(a0, `docker`, `network`, `rm`, netName)),
		echo("all nodes left swarm"),
	)
}

func addDockerGroup(a At) P {
	pc := a.PC
	pc = experimental.WithLog(pc)
	return pc(`sudo`, `gpasswd`, `-a`, a.User, `docker`)
}

func dockerLogin(a At) P {
	pc := a.PC
	pc = experimental.WithLog(pc)
	return seq(
		// a.PC(`sudo`, `cp`, `-r`, `/home/kungfu/.docker`, `/home/`+a.User),
		pc(`sudo`, `chown`, `-R`, a.User, `/home/`+a.User+`/.docker`),
	)
}

func leaveSwarm(a At) P {
	return seq(
		restartDocker(a),
		ignore(urpc(a, `docker`, `swarm`, `leave`, `--force`)),
	)
}

func restartDocker(a At) P {
	return ignore(urpc(a, `sudo`, `systemctl`, `restart`, `docker`))
}

func initSwarmLeader(a0 At) P {
	return seq(
		urpc(a0, `docker`, `swarm`, `init`, `--advertise-addr`, `155.198.152.18`),
		urpc(a0, `docker`, `network`, `create`, `--scope`, `swarm`, `--driver`, `overlay`, `--attachable`, netName),
	)
}

func joinSwarm(a At, token string, leaderIP string) P {
	return seq(
		urpc(a, `docker`, `swarm`, `join`, `--token`, token, leaderIP+`:2377`),
		echo(a.Host+` joined`),
	)
}

func getJoinToken(a At) P {
	return urpc(a, `docker`, `swarm`, `join-token`, `worker`, `-q`)
}
