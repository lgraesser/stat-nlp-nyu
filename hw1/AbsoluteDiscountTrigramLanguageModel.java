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
 * Absolute discounting for a trigram language model, plus a single
 * ficticious count for unknown words.
 * When last tested, this model produces NaN for the HUB and validation
 * perplexity scores. It has not been tested since the evaluation 
 * mechanism was updated.
 * Code adapted from Slav Petrov's NLP Language model starter code
 */
class AbsoluteDiscountTrigramLanguageModel implements LanguageModel {

	static final String START = "<S>";
	static final String STOP = "</S>";
	static final String UNKNOWN = "*UNKNOWN*";
	static double discount_b;
	static double discount_t;

	Counter<String> wordCounter = new Counter<String>();
	CounterMap<String, String> bigramCounter = new CounterMap<String, String>();
	CounterMap<String, String> trigramCounter = new CounterMap<String, String>();
	CounterMap<String, String> discountedBigramCounter = new CounterMap<String, String>();
	CounterMap<String, String> discountedTrigramCounter = new CounterMap<String, String>();
	Counter<String> lmdaCounter_t = new Counter<String>();
	Counter<String> lmdaCounter_b = new Counter<String>();

	public double getTrigramProbability(String prePreviousWord,
			String previousWord, String word) {
		double trigramCount = discountedTrigramCounter.getCount(prePreviousWord + previousWord, word);
		double bigramCount = discountedBigramCounter.getCount(previousWord, word);
		double unigramCount = wordCounter.getCount(word);
		double lambda_b = lmdaCounter_b.getCount(previousWord);
		double lambda_t = lmdaCounter_t.getCount(prePreviousWord + previousWord);
		double total_t = trigramCounter.getCounter(prePreviousWord + previousWord).totalCount();
		double total_b = bigramCounter.getCounter(previousWord).totalCount();

		if (lambda_b > total_b || lambda_t > total_t) {
			System.out.println("Trigram considered is: " + prePreviousWord + " " + 
															previousWord + " " + word);
			System.out.println("total_t: " + total_t);
			System.out.println("total_b: " + total_b);
			System.out.println("lambda_b: " + lambda_b);
			System.out.println("lambda_t: " + lambda_t);
		}

		lambda_b = discount_b * (lambda_b / total_b);
		lambda_t = discount_t * (lambda_t / total_t);
		double result = trigramCount + lambda_t * (bigramCount + lambda_b * unigramCount);

		if (unigramCount == 0) {
			//System.out.println("Unknown word: " + word);
			return wordCounter.getCount(UNKNOWN);
		}
		if (result >= 1) {
			System.out.println("ERROR: Probability > 1");
			System.out.println("Trigram considered is: " + prePreviousWord + " " + 
															previousWord + " " + word);
			System.out.println("lambda_b: " + lambda_b);
			System.out.println("lambda_t: " + lambda_t);
			System.out.println("trigramCount: " + trigramCount);
			System.out.println("bigramCount: " + bigramCount);
			System.out.println("unigramCount: " + unigramCount);

		}
		return result;
	}

	public double getSentenceProbability(List<String> sentence) {
		List<String> stoppedSentence = new ArrayList<String>(sentence);
		stoppedSentence.add(0, START);
		stoppedSentence.add(0, START);
		stoppedSentence.add(STOP);
		double probability = 1.0;
		String prePreviousWord = stoppedSentence.get(0);
		String previousWord = stoppedSentence.get(1);
		//System.out.println("Beginning of sentence: " + prePreviousWord + " " + previousWord);
		for (int i = 2; i < stoppedSentence.size(); i++) {
			String word = stoppedSentence.get(i);
			probability *= getTrigramProbability(prePreviousWord, previousWord,
					word);
			//System.out.println("Word is: " + word + " and new prob. is: " + probability);
			//System.out.println("Trigram probability is: " + 
			//			getTrigramProbability(prePreviousWord, previousWord, word));
			prePreviousWord = previousWord;
			previousWord = word;
		}
		//if (Double.isNaN(probability)) {
		//	System.out.println("Sentence has Nan probability: " + probability);
		//}
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

	public AbsoluteDiscountTrigramLanguageModel(
			Collection<List<String>> sentenceCollection, 
			double discount_b_input,
			double discount_t_input) {
		System.out.println("AbsoluteDiscountTrigramLanguageModel");
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
				lmdaCounter_b.incrementCount(previousWord, 1.0);
			}
		}
		discount_b = n1 / (n1 + 2 * n2);
		System.out.println("Estimated ideal discount is: " + discount_b);
		discount_b = discount_b_input;
		System.out.println("Actual discount is: " + discount_b);

		for (String previousWord : bigramCounter.keySet()) {
			Counter<String> currentCounter = bigramCounter.getCounter(previousWord);
			for (Map.Entry<String, Double> entry : currentCounter.getEntrySet()) {
				Double count = entry.getValue();
				String word = entry.getKey();
				Double newCount = Math.max(count - discount_b, 0);
				discountedBigramCounter.incrementCount(previousWord, word, newCount);
			}
		}

		n1 = 0;
		n2 = 0;
		for (String previousWords : trigramCounter.keySet()) {
			Counter<String> currentCounter = trigramCounter.getCounter(previousWords);
			for (Map.Entry<String, Double> entry : currentCounter.getEntrySet()) {
				Double count = entry.getValue();
				String word = entry.getKey();
				if (count == 1) {
					n1++;
				}
				if (count == 2) {
					n2++;
				}
				lmdaCounter_t.incrementCount(previousWords, 1.0);
			}
		}
		discount_t = n1 / (n1 + 2 * n2);
		System.out.println("Estimated trigram ideal discount is: " + discount_t);
		discount_t = discount_t_input;
		System.out.println("Actual discount is: " + discount_t);

		for (String previousWords : trigramCounter.keySet()) {
			Counter<String> currentCounter = trigramCounter.getCounter(previousWords);
			for (Map.Entry<String, Double> entry : currentCounter.getEntrySet()) {
				Double count = entry.getValue();
				String word = entry.getKey();
				Double newCount = Math.max(count - discount_t, 0);
				discountedTrigramCounter.incrementCount(previousWords, word, newCount);
			}
		}
		normalizeDistributions();
	}

	private void normalizeDistributions() {
		for (String previousWord : discountedBigramCounter.keySet()) {
			Counter<String> currentCounter = discountedBigramCounter.getCounter(previousWord);
			double total = bigramCounter.getCounter(previousWord).totalCount();
			for (Map.Entry<String, Double> entry : currentCounter.getEntrySet()) {
				Double count = entry.getValue();
				String word = entry.getKey();
				Double newCount = count / total;
				discountedBigramCounter.setCount(previousWord, word, newCount);
				if (newCount >= 1) {
					System.out.println("Previous word: " + previousWord);
					System.out.println("Word: " + word);
					System.out.println("Discounted Count: " + count);
					System.out.println("Total Count: " + total);
					System.out.println("New Count: " + newCount);
				}
			}
		}
		for (String previousWords : discountedTrigramCounter.keySet()) {
			Counter<String> currentCounter = discountedTrigramCounter.getCounter(previousWords);
			double total = trigramCounter.getCounter(previousWords).totalCount();
			for (Map.Entry<String, Double> entry : currentCounter.getEntrySet()) {
				Double count = entry.getValue();
				String word = entry.getKey();
				Double newCount = count / total;
				discountedTrigramCounter.setCount(previousWords, word, newCount);
				if (newCount >= 1) {
					System.out.println("Previous words: " + previousWords);
					System.out.println("Word: " + word);
					System.out.println("Discounted Count: " + count);
					System.out.println("Total Count: " + total);
					System.out.println("New Count: " + newCount);
				}
			}
		}
		wordCounter.normalize();
	}
}
