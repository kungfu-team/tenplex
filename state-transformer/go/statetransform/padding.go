package statetransform

func VocabSizePadding(vocabSize int, mpSize int) int {
	makeVocabSizeDivisibleBy := 128
	after := vocabSize
	multiple := makeVocabSizeDivisibleBy * mpSize
	for {
		if after%multiple != 0 {
			after += 1
		} else {
			break
		}
	}
	return after
}
