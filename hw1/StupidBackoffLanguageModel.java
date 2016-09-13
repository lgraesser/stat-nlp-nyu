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
 * Implements stupid backoff for a trigram language model, plus number of unknown words
 * equal to number of words that appear once in the training data
 * Code adapted from Slav Petrov's NLP Language model starter code
 */
class StupidBackoffLanguageModel implements LanguageModel {

	static final String START = "<S>";
	static final String STOP = "</S>";
	static final String UNKNOWN = "*UNKNOWN*";
	static final double EPSILON = 0.000000001;
	static double backoff_b;
	static double backoff_u;
	static double backoff_unk;

	Counter<String> wordCounter = new Counter<String>();
	CounterMap<String, String> bigramCounter = new CounterMap<String, String>();
	CounterMap<String, String> trigramCounter = new CounterMap<String, String>();

	public double getTrigramProbability(String prePreviousWord,
			String previousWord, String word) {
		double trigramCount = trigramCounter.getCount(prePreviousWord
				+ previousWord, word);
		double bigramCount = bigramCounter.getCount(previousWord, word);
		double unigramCount = wordCounter.getCount(word);

		if (!(trigramCount == 0)) {
			return trigramCount;
		}
		else if (!(bigramCount == 0)) {
			return bigramCount * backoff_b;
		}
		else if (!(unigramCount == 0)) {
			return unigramCount * backoff_u;
		}
		else {
			//System.out.println("Unknown word: " + word);
			return wordCounter.getCount(UNKNOWN) * backoff_unk;
		}
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

	public StupidBackoffLanguageModel(
			Collection<List<String>> sentenceCollection,
			double bigrambackoff, double unigrambackoff,
			double unknownbackoff) {
		backoff_b = bigrambackoff;
		backoff_u = unigrambackoff;
		backoff_unk = unknownbackoff;
		System.out.println("StupidBackoffLanguageModel: bigrambackoff: " + backoff_b +
							" unigrambackoff: " + backoff_u + " backoff_unk: " + backoff_unk);
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
				wordCounter.incrementCount(word, 1.0);
				bigramCounter.incrementCount(previousWord, word, 1.0);
				trigramCounter.incrementCount(prePreviousWord + previousWord,
						word, 1.0);
				prePreviousWord = previousWord;
				previousWord = word;
			}
		}
		// Set unknown word count to the total number of words that appeared once in this data
		for (Map.Entry<String, Double> entry : wordCounter.getEntrySet()) {
			double count = entry.getValue();
			String word = entry.getKey();
			if (count == 1) {
				unknownWords++;
			}
		}

		System.out.println("Number of unknown words: " + unknownWords);
		wordCounter.incrementCount(UNKNOWN, unknownWords);
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
