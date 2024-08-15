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

func isIntAndCheck(arr []string, key int) bool {
	if key >= len(arr) {
		return false
	}
	_, err := strconv.Atoi(arr[key])
	return err == nil
}

func equalAndCheck[T int | string](arr []T, key int, val T) bool {
	if key >= len(arr) {
		return false
	}
	return arr[key] == val
}
