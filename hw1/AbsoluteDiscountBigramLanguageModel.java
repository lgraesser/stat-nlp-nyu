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
 * Absolute discounting for a bigram language model plus a single
 * ficticious count for unknown words.
 * Code adapted from Slav Petrov's NLP Language model starter code
 */
class AbsoluteDiscountBigramLanguageModel implements LanguageModel {

	static final String START = "<S>";
	static final String STOP = "</S>";
	static final String UNKNOWN = "*UNKNOWN*";
	static double discount;

	Counter<String> wordCounter = new Counter<String>();
	CounterMap<String, String> bigramCounter = new CounterMap<String, String>();
	CounterMap<String, String> discountedBigramCounter = new CounterMap<String, String>();
	Counter<String> lmdaCounter = new Counter<String>();

	public double getBigramProbability(String previousWord, String word) {
		double bigramCount = discountedBigramCounter.getCount(previousWord, word);
		double unigramCount = wordCounter.getCount(word);
		double lambda = lmdaCounter.getCount(previousWord);
		double result = bigramCount + lambda * unigramCount;
		if (result == 0) {
			return wordCounter.getCount(UNKNOWN);
		}
		return result;
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

	public AbsoluteDiscountBigramLanguageModel(
			Collection<List<String>> sentenceCollection, double discount_input) {
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
		wordCounter.incrementCount(UNKNOWN, 1.0);

		double n1 = 0;
		double n2 = 0;
		for (String previousWord : bigramCounter.keySet()) {
			Counter<String> currentCounter = bigramCounter.getCounter(previousWord);
			for (Map.Entry<String, Double> entry : currentCounter.getEntrySet()) {
				Double count = entry.getValue();
				String word = entry.getKey();
				if (count == 1) {
					n1++;
				}
				if (count == 2) {
					n2++;
				}
				lmdaCounter.incrementCount(previousWord, 1.0);
			}
		}
		discount = n1 / (n1 + 2 * n2);
		System.out.println("Estimated ideal discount is: " + discount);
		discount = discount_input;
		System.out.println("Actual discount is: " + discount);

		for (String previousWord : bigramCounter.keySet()) {
			Counter<String> currentCounter = bigramCounter.getCounter(previousWord);
			for (Map.Entry<String, Double> entry : currentCounter.getEntrySet()) {
				Double count = entry.getValue();
				String word = entry.getKey();
				Double newCount = Math.max(count - discount, 0);
				discountedBigramCounter.incrementCount(previousWord, word, newCount);
			}
		}

		normalizeDistributions();
	}

	private void normalizeDistributions() {
		for (String previousWord : discountedBigramCounter.keySet()) {
			Counter<String> currentCounter = discountedBigramCounter.getCounter(previousWord);
			double total = bigramCounter.getCounter(previousWord).totalCount();
			currentCounter.scale(1.0 / total);
		}
		for (String previousWord : lmdaCounter.keySet()) {
			double total = bigramCounter.getCounter(previousWord).totalCount();
			double count = lmdaCounter.getCount(previousWord);
			double newCount = discount / total * count;
			lmdaCounter.setCount(previousWord, newCount);
		}
		for (String previousWord : bigramCounter.keySet()) {
			bigramCounter.getCounter(previousWord).normalize();
		}
		wordCounter.normalize();
	}
}
