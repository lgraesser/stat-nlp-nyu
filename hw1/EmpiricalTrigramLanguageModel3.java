package nlp.assignments;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;

/**
 * Interpolated trigram language model with the capacity to account for UNK by replacing
 * the first instance of each word with UNK. It is possible to change the types of words
 * to which this applies based on the frequency of a word in the training set
 * Code adapted from Slav Petrov's NLP Language model starter code
 */
class EmpiricalTrigramLanguageModel3 implements LanguageModel {

	static final String START = "<S>";
	static final String STOP = "</S>";
	static final String UNKNOWN = "*UNKNOWN*";
	static final double lambda1 = 0.5;
	static final double lambda2 = 0.3;

	Counter<String> wordCounter = new Counter<String>();
	CounterMap<String, String> bigramCounter = new CounterMap<String, String>();
	CounterMap<String, String> trigramCounter = new CounterMap<String, String>();

	public double getTrigramProbability(String prePreviousWord,
			String previousWord, String word) {
		double trigramCount = trigramCounter.getCount(prePreviousWord
				+ previousWord, word);
		double bigramCount = bigramCounter.getCount(previousWord, word);
		double unigramCount = wordCounter.getCount(word);
		if (unigramCount == 0) {
			//System.out.println("UNKNOWN Word: " + word);
			unigramCount = wordCounter.getCount(UNKNOWN);
		}
		return lambda1 * trigramCount + lambda2 * bigramCount
				+ (1.0 - lambda1 - lambda2) * unigramCount;
	}

	public double getSentenceProbability(List<String> sentence) {
		List<String> stoppedSentence = new ArrayList<String>(sentence);
		stoppedSentence.add(0, START);
		stoppedSentence.add(0, START);
		stoppedSentence.add(STOP);
		double probability = 1.0;
		String prePreviousWord = stoppedSentence.get(0);
		String previousWord = stoppedSentence.get(1);
		for (int i = 2; i < stoppedSentence.size(); i++) {
			String word = stoppedSentence.get(i);
			probability *= getTrigramProbability(prePreviousWord, previousWord,
					word);
			prePreviousWord = previousWord;
			previousWord = word;
		}
		return probability;
	}

	String generateWord() {
		double sample = Math.random();
		double sum = 0.0;
		for (String word : wordCounter.keySet()) {
			sum += wordCounter.getCount(word);
			if (sum > sample) {
				return word;
			}
		}
		return UNKNOWN;
	}

	// Word generator conditioned on previous word, uses bigram counter
	String generateWord(String previousWord) {
		double sample = Math.random();
		double sum = 0.0;
		for (String word : wordCounter.keySet()) {
			sum += bigramCounter.getCount(previousWord, word);
			if (sum > sample) {
				return word;
			}
		}
		return UNKNOWN;
	}

	// Word generator conditioned on two previous words, uses trigram counter
	String generateWord(String prePreviousWord, String previousWord) {
		double sample = Math.random();
		double sum = 0.0;
		for (String word : wordCounter.keySet()) {
			sum += trigramCounter.getCount(prePreviousWord + previousWord, word);
			if (sum > sample) {
				return word;
			}
		}
		return UNKNOWN;
	}

	// Adapted to makes use of bigram and trigram probabilities
	public List<String> generateSentence() {
		List<String> sentence = new ArrayList<String>();
		String prePreviousWord = generateWord();
		sentence.add(prePreviousWord);
		if (prePreviousWord.equals(STOP)) {
			return sentence;
		}
		String previousWord = generateWord(prePreviousWord);
		sentence.add(previousWord);
		if (previousWord.equals(STOP)) {
			return sentence;
		}
		
		// Often sentences don't terminate, cap on sentence length to ensure program ends
		String word = generateWord(prePreviousWord, previousWord);
		while (!word.equals(STOP) && sentence.size() <= 30) {
			sentence.add(word);
			prePreviousWord = previousWord;
			previousWord = word;
			word = generateWord(prePreviousWord, previousWord);
		}
		return sentence;
	}

	public EmpiricalTrigramLanguageModel3(
			Collection<List<String>> sentenceCollection) {
		int unknownWords = 0;
		for (List<String> sentence : sentenceCollection) {
			List<String> stoppedSentence = new ArrayList<String>(sentence);
			stoppedSentence.add(0, START);
			stoppedSentence.add(0, START);
			stoppedSentence.add(STOP);
			String prePreviousWord = stoppedSentence.get(0);
			String previousWord = stoppedSentence.get(1);
			for (int i = 2; i < stoppedSentence.size(); i++) {
				String word = stoppedSentence.get(i);
				// Treating first occurrence of each word as unknown, only for unigrams
				// Assumes that many bigrams and trigrams will be unknown, penalising this too strong
				if (!wordCounter.containsKey(word)) {
					wordCounter.setCount(word, 0);
					unknownWords++;
				}
				else {
					wordCounter.incrementCount(word, 1.0);
				}
				bigramCounter.incrementCount(previousWord, word, 1.0);
				trigramCounter.incrementCount(prePreviousWord + previousWord,
						word, 1.0);
				prePreviousWord = previousWord;
				previousWord = word;
			}
		}
		// Remove unknown count from common words (occurred >= x times) and adding back onto original word count
		for (Map.Entry<String, Double> entry : wordCounter.getEntrySet()) {
			double count = entry.getValue();
			String word = entry.getKey();
			if (count >= 7) {
				//unknownWords--;
				//wordCounter.incrementCount(word, 1.0);
			}
		}

		System.out.println("Number of unknown words: " + unknownWords);
		wordCounter.incrementCount(UNKNOWN, 1);
		normalizeDistributions();
	}

	private void normalizeDistributions() {
		for (String previousBigram : trigramCounter.keySet()) {
			trigramCounter.getCounter(previousBigram).normalize();
		}
		for (String previousWord : bigramCounter.keySet()) {
			bigramCounter.getCounter(previousWord).normalize();
		}
		wordCounter.normalize();
	}
}
