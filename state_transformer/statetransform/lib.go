package statetransform

import "strconv"

func equal[T int | string](a, b []T) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

func isInt(str string) bool {
	_, err := strconv.Atoi(str)
	return err == nil
}
