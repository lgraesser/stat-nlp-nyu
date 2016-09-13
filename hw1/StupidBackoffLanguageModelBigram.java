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
 * Implements stupid backoff for a bigram language model, plus number of unknown words
 * equal to number of words that appear once in the training data
 * Code adapted from Slav Petrov's NLP Language model starter code
 */
class StupidBackoffLanguageModelBigram implements LanguageModel {

	static final String START = "<S>";
	static final String STOP = "</S>";
	static final String UNKNOWN = "*UNKNOWN*";
	static double backoff_u;
	static double backoff_unk;

	Counter<String> wordCounter = new Counter<String>();
	CounterMap<String, String> bigramCounter = new CounterMap<String, String>();

	public double getBigramProbability(String previousWord, String word) {
		double bigramCount = bigramCounter.getCount(previousWord, word);
		double unigramCount = wordCounter.getCount(word);
		if (!(bigramCount == 0)) {
			return bigramCount;
		}
		else if (!(unigramCount == 0)) {
			return unigramCount * backoff_u;
		}
		else {
			return wordCounter.getCount(UNKNOWN) * backoff_unk;
		}
	}

	public double getSentenceProbability(List<String> sentence) {
		List<String> stoppedSentence = new ArrayList<String>(sentence);
		stoppedSentence.add(0, START);
		stoppedSentence.add(STOP);
		double probability = 1.0;
		String previousWord = stoppedSentence.get(0);
		for (int i = 1; i < stoppedSentence.size(); i++) {
			String word = stoppedSentence.get(i);
			probability *= getBigramProbability(previousWord, word);
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

	// Adapted to makes use of bigram probabilities
	public List<String> generateSentence() {
		List<String> sentence = new ArrayList<String>();
		String previousWord = generateWord();
		sentence.add(previousWord);
		if (previousWord.equals(STOP)) {
			return sentence;
		}
		String word = generateWord(previousWord);
		while (!word.equals(STOP) && sentence.size() <= 30) {
			sentence.add(word);
			previousWord = word;
			word = generateWord(previousWord);
		}
		return sentence;
	}

	public StupidBackoffLanguageModelBigram(
			Collection<List<String>> sentenceCollection, 
			double unigrambackoff, double unknownbackoff) {
		int unknownWords = 0;
		backoff_u = unigrambackoff;
		backoff_unk = unknownbackoff;
		System.out.println("StupidBackoffLanguageModelBigram: unigrambackoff: " + unigrambackoff +
			" unknownbackoff: " + unknownbackoff);
		for (List<String> sentence : sentenceCollection) {
			List<String> stoppedSentence = new ArrayList<String>(sentence);
			stoppedSentence.add(0, START);
			stoppedSentence.add(STOP);
			String previousWord = stoppedSentence.get(0);
			for (int i = 1; i < stoppedSentence.size(); i++) {
				String word = stoppedSentence.get(i);
				wordCounter.incrementCount(word, 1.0);
				bigramCounter.incrementCount(previousWord, word, 1.0);
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
		for (String previousWord : bigramCounter.keySet()) {
			bigramCounter.getCounter(previousWord).normalize();
		}
		wordCounter.normalize();
	}
}
