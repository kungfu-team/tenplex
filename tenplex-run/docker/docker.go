package docker

import (
	"fmt"
	"strings"
)

func SetupSwarm(admin string, ips []string, network string) P {
	if len(ips) <= 0 {
		return echo("no workers")
	}
	as := fmap(func(h string) At { return at(admin, h) }, ips...)
	a0 := as[0]
	return seq(
		parmap(addDockerGroup, as...),
		parmap(dockerLogin, as...),
		cleanupSwarm(admin, ips, network),
		initSwarmLeader(a0, ips[0], network),
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

func cleanupSwarm(admin string, pubIPs []string, network string) P {
	as := fmap(func(h string) At { return at(admin, h) }, pubIPs...)
	a0 := as[0]
	return seq(
		parmap(leaveSwarm, as...),
		ignore(urpc(a0, `docker`, `network`, `rm`, network)),
		echo("all nodes left swarm"),
	)
}

func addDockerGroup(a At) P {
	return a.PC(`sudo`, `gpasswd`, `-a`, a.User, `docker`)
}

func dockerLogin(a At) P {
	return seq(
		a.PC(`sudo`, `chown`, `-R`, a.User, `/home/`+a.User+`/.docker`),
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

func initSwarmLeader(a0 At, ip string, network string) P {
	return seq(
		urpc(a0, `docker`, `swarm`, `init`, `--advertise-addr`, ip),
		urpc(a0, `docker`, `network`, `create`, `--scope`, `swarm`, `--driver`, `overlay`, `--attachable`, network),
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
