package nlp.assignments;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.text.NumberFormat;
import java.text.DecimalFormat;

import nlp.langmodel.LanguageModel;
import nlp.util.CommandLineUtils;

/**
 * Code has been adapted from Slav Petrov's NLP Language model harness
 * The main change has been to add parameter grid search for a number of models
 * and to print the evaluation of a number of models to the screen at once
 *
 * This is the main harness for assignment 1. To run this harness, use
 * <p/>
 * java nlp.assignments.LanguageModelTester -path ASSIGNMENT_DATA_PATH -model
 * MODEL_DESCRIPTOR_STRING
 * <p/>
 * First verify that the data can be read on your system. Second, find the point
 * in the main method (near the bottom) where an EmpiricalUnigramLanguageModel
 * is constructed. You will be writing new implementations of the LanguageModel
 * interface and constructing them there.
 */
public class LanguageModelTester3 {

	// HELPER CLASS FOR THE HARNESS, CAN IGNORE
	static class EditDistance {
		static double INSERT_COST = 1.0;	
		static double DELETE_COST = 1.0;
		static double SUBSTITUTE_COST = 1.0;

		private double[][] initialize(double[][] d) {
			for (int i = 0; i < d.length; i++) {
				for (int j = 0; j < d[i].length; j++) {
					d[i][j] = Double.NaN;
				}
			}
			return d;
		}

		public double getDistance(List<? extends Object> firstList,
				List<? extends Object> secondList) {
			double[][] bestDistances = initialize(new double[firstList.size() + 1][secondList
					.size() + 1]);
			return getDistance(firstList, secondList, 0, 0, bestDistances);
		}

		private double getDistance(List<? extends Object> firstList,
				List<? extends Object> secondList, int firstPosition,
				int secondPosition, double[][] bestDistances) {
			if (firstPosition > firstList.size()
					|| secondPosition > secondList.size())
				return Double.POSITIVE_INFINITY;
			if (firstPosition == firstList.size()
					&& secondPosition == secondList.size())
				return 0.0;
			if (Double.isNaN(bestDistances[firstPosition][secondPosition])) {
				double distance = Double.POSITIVE_INFINITY;
				distance = Math.min(
						distance,
						INSERT_COST
								+ getDistance(firstList, secondList,
										firstPosition + 1, secondPosition,
										bestDistances));
				distance = Math.min(
						distance,
						DELETE_COST
								+ getDistance(firstList, secondList,
										firstPosition, secondPosition + 1,
										bestDistances));
				distance = Math.min(
						distance,
						SUBSTITUTE_COST
								+ getDistance(firstList, secondList,
										firstPosition + 1, secondPosition + 1,
										bestDistances));
				if (firstPosition < firstList.size()
						&& secondPosition < secondList.size()) {
					if (firstList.get(firstPosition).equals(
							secondList.get(secondPosition))) {
						distance = Math.min(
								distance,
								getDistance(firstList, secondList,
										firstPosition + 1, secondPosition + 1,
										bestDistances));
					}
				}
				bestDistances[firstPosition][secondPosition] = distance;
			}
			return bestDistances[firstPosition][secondPosition];
		}
	}

	// HELPER CLASS FOR THE HARNESS, CAN IGNORE
	static class SentenceCollection extends AbstractCollection<List<String>> {
		static class SentenceIterator implements Iterator<List<String>> {

			BufferedReader reader;

			public boolean hasNext() {
				try {
					return reader.ready();
				} catch (IOException e) {
					return false;
				}
			}

			public List<String> next() {
				try {
					String line = reader.readLine();
					String[] words = line.split("\\s+");
					List<String> sentence = new ArrayList<String>();
					for (int i = 0; i < words.length; i++) {
						String word = words[i];
						sentence.add(word.toLowerCase());
					}
					return sentence;
				} catch (IOException e) {
					throw new NoSuchElementException();
				}
			}

			public void remove() {
				throw new UnsupportedOperationException();
			}

			public SentenceIterator(BufferedReader reader) {
				this.reader = reader;
			}
		}

		String fileName;

		public Iterator<List<String>> iterator() {
			try {
				BufferedReader reader = new BufferedReader(new FileReader(
						fileName));
				return new SentenceIterator(reader);
			} catch (FileNotFoundException e) {
				throw new RuntimeException("Problem with SentenceIterator for "
						+ fileName);
			}
		}

		public int size() {
			int size = 0;
			Iterator<List<String>> i = iterator();
			while (i.hasNext()) {
				size++;
				i.next();
			}
			return size;
		}

		public SentenceCollection(String fileName) {
			this.fileName = fileName;
		}

		public static class Reader {
			static Collection<List<String>> readSentenceCollection(
					String fileName) {
				return new SentenceCollection(fileName);
			}
		}

	}

	static double calculatePerplexity(LanguageModel languageModel,
			Collection<List<String>> sentenceCollection) {
		double logProbability = 0.0;
		double numSymbols = 0.0;
		for (List<String> sentence : sentenceCollection) {
			logProbability += Math.log(languageModel
					.getSentenceProbability(sentence)) / Math.log(2.0);
			numSymbols += sentence.size();
		}
		double avgLogProbability = logProbability / numSymbols;
		double perplexity = Math.pow(0.5, avgLogProbability);
		return perplexity;
	}

	static double calculateWordErrorRate(LanguageModel languageModel,
			List<SpeechNBestList> speechNBestLists, boolean verbose) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			List<String> bestGuess = null;
			double bestScore = Double.NEGATIVE_INFINITY;
			double numWithBestScores = 0.0;
			double distanceForBestScores = 0.0;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double score = Math.log(languageModel
						.getSentenceProbability(guess))
						+ (speechNBestList.getAcousticScore(guess) / 16.0);
				double distance = editDistance.getDistance(correctSentence,
						guess);
				if (score == bestScore) {
					numWithBestScores += 1.0;
					distanceForBestScores += distance;
				}
				if (score > bestScore || bestGuess == null) {
					bestScore = score;
					bestGuess = guess;
					distanceForBestScores = distance;
					numWithBestScores = 1.0;
				}
			}
			// double distance = editDistance.getDistance(correctSentence,
			// bestGuess);
			totalDistance += distanceForBestScores / numWithBestScores;
			totalWords += correctSentence.size();
			if (verbose) {
				System.out.println();
				displayHypothesis("GUESS:", bestGuess, speechNBestList,
						languageModel);
				displayHypothesis("GOLD:", correctSentence, speechNBestList,
						languageModel);
			}
		}
		return totalDistance / totalWords;
	}

	private static NumberFormat nf = new DecimalFormat("0.00E00");

	private static void displayHypothesis(String prefix, List<String> guess,
			SpeechNBestList speechNBestList, LanguageModel languageModel) {
		double acoustic = speechNBestList.getAcousticScore(guess) / 16.0;
		double language = Math.log(languageModel.getSentenceProbability(guess));
		System.out.println(prefix + "\tAM: " + nf.format(acoustic) + "\tLM: "
				+ nf.format(language) + "\tTotal: "
				+ nf.format(acoustic + language) + "\t" + guess);
	}

	static double calculateWordErrorRateLowerBound(
			List<SpeechNBestList> speechNBestLists) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			double bestDistance = Double.POSITIVE_INFINITY;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double distance = editDistance.getDistance(correctSentence,
						guess);
				if (distance < bestDistance)
					bestDistance = distance;
			}
			totalDistance += bestDistance;
			totalWords += correctSentence.size();
		}
		return totalDistance / totalWords;
	}

	static double calculateWordErrorRateUpperBound(
			List<SpeechNBestList> speechNBestLists) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			double worstDistance = Double.NEGATIVE_INFINITY;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double distance = editDistance.getDistance(correctSentence,
						guess);
				if (distance > worstDistance)
					worstDistance = distance;
			}
			totalDistance += worstDistance;
			totalWords += correctSentence.size();
		}
		return totalDistance / totalWords;
	}

	static double calculateWordErrorRateRandomChoice(
			List<SpeechNBestList> speechNBestLists) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			double sumDistance = 0.0;
			double numGuesses = 0.0;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double distance = editDistance.getDistance(correctSentence,
						guess);
				sumDistance += distance;
				numGuesses += 1.0;
			}
			totalDistance += sumDistance / numGuesses;
			totalWords += correctSentence.size();
		}
		return totalDistance / totalWords;
	}

	static Collection<List<String>> extractCorrectSentenceList(
			List<SpeechNBestList> speechNBestLists) {
		Collection<List<String>> correctSentences = new ArrayList<List<String>>();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			correctSentences.add(speechNBestList.getCorrectSentence());
		}
		return correctSentences;
	}

	static Set<String> extractVocabulary(
			Collection<List<String>> sentenceCollection) {
		Set<String> vocabulary = new HashSet<String>();
		for (List<String> sentence : sentenceCollection) {
			for (String word : sentence) {
				vocabulary.add(word);
			}
		}
		return vocabulary;
	}

	public static void main(String[] args) throws IOException {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		String model = "baseline";
		boolean verbose = false;

		// Update defaults using command line specifications

		// The path to the assignment data
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
		}
		System.out.println("Using base path: " + basePath);

		// A string descriptor of the model to use
		if (argMap.containsKey("-model")) {
			model = argMap.get("-model");
		}
		System.out.println("Using model: " + model);

		// Whether or not to print the individual speech errors.
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}
		if (argMap.containsKey("-quiet")) {
			verbose = false;
		}

		// Read in all the assignment data
		String trainingSentencesFile = "/treebank-sentences-spoken-train.txt";
		String speechNBestListsPath = "/wsj_n_bst";
		Collection<List<String>> trainingSentenceCollection = SentenceCollection.Reader
				.readSentenceCollection(basePath + trainingSentencesFile);
		Set<String> trainingVocabulary = extractVocabulary(trainingSentenceCollection);
		List<SpeechNBestList> speechNBestLists = SpeechNBestList.Reader
				.readSpeechNBestLists(basePath + speechNBestListsPath,
						trainingVocabulary);

		String validationSentencesFile = "/treebank-sentences-spoken-validate.txt";
		Collection<List<String>> validationSentenceCollection =
			SentenceCollection.Reader.readSentenceCollection(basePath + validationSentencesFile);
		Set<String> validationVocabulary = extractVocabulary(trainingSentenceCollection);

		String testSentencesFile = "/treebank-sentences-spoken-test.txt";
		Collection<List<String>> testSentenceCollection =
			SentenceCollection.Reader.readSentenceCollection(basePath + testSentencesFile);

		// Build the language model
		LanguageModel languageModel = null;
		if (model.equalsIgnoreCase("baseline")) {
			languageModel = new EmpiricalUnigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("sri")) {
			languageModel = new SriLanguageModel(argMap.get("-sri"));
		} else if (model.equalsIgnoreCase("bigram")) {
			languageModel = new EmpiricalBigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("trigram")) {
			languageModel = new EmpiricalTrigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("trigram_LG")) {
			languageModel = new EmpiricalTrigramLanguageModel3(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("katz-bigram")) {
			languageModel = new KatzBigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("katz-trigram")) {
			languageModel = new KatzTrigramLanguageModel(
					trainingSentenceCollection);
		} else {
			throw new RuntimeException("Unknown model descriptor: " + model);
		}

		System.out.println("Evaluating baseline model");
		evaluateModel(languageModel, speechNBestLists, trainingSentenceCollection,
					   validationSentenceCollection, testSentenceCollection,
					   verbose);
		System.out.println();

		// Evaluate flags
		boolean evaluate_stupid_bigram = false;
		boolean evaluate_stupid_trigram = false;
		boolean evaluate_abs_bigram = false;
		boolean evaluate_abs_trigram = false;
		int k = 0;
		int r = 0;
		
		if (evaluate_stupid_bigram) {
			// Grid search for Stupid backoff bigram model optimal parameters
			double[][] sb_bigram_train_results = new double[10][10];
			double[][] sb_bigram_valid_results = new double[10][10];
			double[][] sb_bigram_hub_results = new double[10][10];
			double[][] sb_bigram_wer_results = new double[10][10];
			k = 0;
			r = 0;
			for (double i = 0.1; i < 1.0; i += 0.1) {
				System.out.println("Processing row " + i + " of SB bigram LM");
				r = 0;
				for (double j = 0.1; j < 1.0; j += 0.1) {
					languageModel = new StupidBackoffLanguageModelBigram(
							trainingSentenceCollection, i, j);
					List<Double> results = evaluateModel(languageModel, speechNBestLists, trainingSentenceCollection,
							   validationSentenceCollection, testSentenceCollection,
							   verbose);
					sb_bigram_train_results[k][r] = results.get(0);
					sb_bigram_valid_results[k][r] = results.get(1);
					sb_bigram_hub_results[k][r] = results.get(3);
					sb_bigram_wer_results[k][r] = results.get(4);
					System.out.println();
					r++;
				}
				k++;
			}
			System.out.println("Stupid Backoff Bigram Training Perplexity");
			printArray(sb_bigram_train_results);
			System.out.println("Stupid Backoff Bigram Valid Perplexity");
			printArray(sb_bigram_valid_results);
			System.out.println("Stupid Backoff Bigram Hub Perplexity");
			printArray(sb_bigram_hub_results);
			System.out.println("Stupid Backoff Bigram WER");
			printArray(sb_bigram_wer_results);
		}

		if (evaluate_stupid_trigram) {
			// Grid search for Stupid backoff trigram model optimal parameters
			double[][] sb_trigram_train_results = new double[10][10];
			double[][] sb_trigram_valid_results = new double[10][10];
			double[][] sb_trigram_hub_results = new double[10][10];
			double[][] sb_trigram_wer_results = new double[10][10];
			k = 0;
			r = 0;
			for (double i = 0.1; i < 1.0; i += 0.1) {
				System.out.println("Processing row " + i + " of SB trigram LM");
				r = 0;
				for (double j = 0.1; j < 1.0; j += 0.1) {
					languageModel = new StupidBackoffLanguageModel(
							trainingSentenceCollection, i, j, 0.2);
					List<Double> results = evaluateModel(languageModel, speechNBestLists, trainingSentenceCollection,
							   validationSentenceCollection, testSentenceCollection,
							   verbose);
					sb_trigram_train_results[k][r] = results.get(0);
					sb_trigram_valid_results[k][r] = results.get(1);
					sb_trigram_hub_results[k][r] = results.get(3);
					sb_trigram_wer_results[k][r] = results.get(4);
					System.out.println();
					r++;
				}
				k++;
			}

			System.out.println("Stupid Backoff Trigram Training Perplexity");
			printArray(sb_trigram_train_results);
			System.out.println("Stupid Backoff Trigram Valid Perplexity");
			printArray(sb_trigram_valid_results);
			System.out.println("Stupid Backoff Trigram Hub Perplexity");
			printArray(sb_trigram_hub_results);
			System.out.println("Stupid Backoff Trigram WER");
			printArray(sb_trigram_wer_results);
		}

		if (evaluate_abs_bigram) {
			// Grid search for absolute bigram model optimal parameters
			double[] abs_bigram_train_results = new double[10];
			double[] abs_bigram_valid_results = new double[10];
			double[] abs_bigram_hub_results = new double[10];
			double[] abs_bigram_wer_results = new double[10];
			int i = 0;
			for (double j = 0.1; j < 1.0; j += 0.1) {
				languageModel = new AbsoluteDiscountBigramLanguageModel(
							trainingSentenceCollection, j);
				List<Double> results = evaluateModel(languageModel, speechNBestLists, trainingSentenceCollection,
						   validationSentenceCollection, testSentenceCollection,
						   verbose);
				abs_bigram_train_results[i] = results.get(0);
				abs_bigram_valid_results[i] = results.get(1);
				abs_bigram_hub_results[i] = results.get(3);
				abs_bigram_wer_results[i] = results.get(4);
				System.out.println();
				i++;
			}
			System.out.println("Absolute Bigram Training Perplexity");
			printArray(abs_bigram_train_results);
			System.out.println("Absolute Bigram Valid Perplexity");
			printArray(abs_bigram_valid_results);
			System.out.println("Absolute Bigram Hub Perplexity");
			printArray(abs_bigram_hub_results);
			System.out.println("Absolute Bigram WER");
			printArray(abs_bigram_wer_results);
		}

		if (evaluate_abs_trigram) {
			// Grid search for Absolute discounted trigram model optimal parameters
			double[][] sb_trigram_train_results = new double[10][10];
			double[][] sb_trigram_valid_results = new double[10][10];
			double[][] sb_trigram_hub_results = new double[10][10];
			double[][] sb_trigram_wer_results = new double[10][10];
			k = 0;
			r = 0;
			for (double i = 0.1; i < 1.0; i += 0.1) {
				System.out.println("Processing row " + i + " of SB trigram LM");
				r = 0;
				for (double j = 0.1; j < 1.0; j += 0.1) {
					languageModel = new AbsoluteDiscountTrigramLanguageModel(
							trainingSentenceCollection, i, j);
					List<Double> results = evaluateModel(languageModel, speechNBestLists, trainingSentenceCollection,
							   validationSentenceCollection, testSentenceCollection,
							   verbose);
					sb_trigram_train_results[k][r] = results.get(0);
					sb_trigram_valid_results[k][r] = results.get(1);
					sb_trigram_hub_results[k][r] = results.get(3);
					sb_trigram_wer_results[k][r] = results.get(4);
					System.out.println();
					r++;
				}
				k++;
			}

			System.out.println("Absolute Discount Trigram Training Perplexity");
			printArray(sb_trigram_train_results);
			System.out.println("Absolute Discount Trigram Valid Perplexity");
			printArray(sb_trigram_valid_results);
			System.out.println("Absolute Discount Trigram Hub Perplexity");
			printArray(sb_trigram_hub_results);
			System.out.println("Absolute Discount Trigram WER");
			printArray(sb_trigram_wer_results);
		}

		System.out.println("Evaluating models with optimal parameters " + 
						"and combined training and validation set");
		String trainValidSentencesFile = "/treebank-sentences-spoken-trainandvalid.txt";
		Collection<List<String>> trainValidSentenceCollection =
			SentenceCollection.Reader.readSentenceCollection(basePath + trainValidSentencesFile);
		Set<String> trainValidVocabulary = extractVocabulary(trainingSentenceCollection);
		
		System.out.println("Baseline: Empirical Unigram Language model");
		languageModel = new EmpiricalUnigramLanguageModel(trainingSentenceCollection);
		evaluateModelPrintToScreen(languageModel, speechNBestLists,
						trainingSentenceCollection,
						validationSentenceCollection, 
						trainValidSentenceCollection, 
						testSentenceCollection,
						verbose);
		System.out.println();

		System.out.println("Baseline: Empirical Bigram Language model");
		languageModel = new EmpiricalBigramLanguageModel(trainingSentenceCollection);
		evaluateModelPrintToScreen(languageModel, speechNBestLists,
						trainingSentenceCollection,
						validationSentenceCollection, 
						trainValidSentenceCollection, 
						testSentenceCollection,
						verbose);
		System.out.println();

		System.out.println("Baseline: Empirical Trigram Language model");
		languageModel = new EmpiricalTrigramLanguageModel(trainingSentenceCollection);
		evaluateModelPrintToScreen(languageModel, speechNBestLists,
						trainingSentenceCollection,
						validationSentenceCollection, 
						trainValidSentenceCollection, 
						testSentenceCollection,
						verbose);
		System.out.println();

		System.out.println("Empirical Trigram Language model: UNK = num words in training set with count = 1");
		languageModel = new EmpiricalTrigramLanguageModel3(trainingSentenceCollection);
		evaluateModelPrintToScreen(languageModel, speechNBestLists,
						trainingSentenceCollection,
						validationSentenceCollection, 
						trainValidSentenceCollection, 
						testSentenceCollection,
						verbose);
		System.out.println();

		System.out.println("Stupid backoff Bigram Language model");
		languageModel = new StupidBackoffLanguageModelBigram(trainingSentenceCollection, 0.1, 0.1);
		evaluateModelPrintToScreen(languageModel, speechNBestLists,
						trainingSentenceCollection,
						validationSentenceCollection, 
						trainValidSentenceCollection, 
						testSentenceCollection,
						verbose);
		System.out.println();

		System.out.println("Stupid backoff Trigram Language model");
		languageModel = new StupidBackoffLanguageModel(trainingSentenceCollection, 0.5, 0.3, 0.2);
		evaluateModelPrintToScreen(languageModel, speechNBestLists,
						trainingSentenceCollection,
						validationSentenceCollection, 
						trainValidSentenceCollection, 
						testSentenceCollection,
						verbose);
		System.out.println();

		System.out.println("Absolute Discount Bigram Language model");
		languageModel = new AbsoluteDiscountBigramLanguageModel(trainingSentenceCollection, 0.4);
		evaluateModelPrintToScreen(languageModel, speechNBestLists,
						trainingSentenceCollection,
						validationSentenceCollection, 
						trainValidSentenceCollection, 
						testSentenceCollection,
						verbose);
		System.out.println();
	}
	
	private static List<Double> evaluateModel(LanguageModel languageModel, 
							   List<SpeechNBestList> speechNBestLists,
							   Collection<List<String>> trainingSentenceCollection,
							   Collection<List<String>> validationSentenceCollection,
							   Collection<List<String>> testSentenceCollection,
							   boolean verbose) {
		double hubPerplexity = calculatePerplexity(languageModel,
				extractCorrectSentenceList(speechNBestLists));
		double wsjPerplexityTrain = calculatePerplexity(languageModel,
				trainingSentenceCollection);
		double wsjPerplexityValid = calculatePerplexity(languageModel,
				validationSentenceCollection);
		double wsjPerplexityTest = calculatePerplexity(languageModel,
				testSentenceCollection);
		double wordErrorRate = calculateWordErrorRate(languageModel,
				speechNBestLists, verbose);

		List<Double> results = new ArrayList<Double>();
		results.add(wsjPerplexityTrain);
		results.add(wsjPerplexityValid);
		results.add(wsjPerplexityTest);
		results.add(hubPerplexity);
		results.add(wordErrorRate);
		return results;
	}

	private static void evaluateModelPrintToScreen(LanguageModel languageModel, 
							   List<SpeechNBestList> speechNBestLists,
							   Collection<List<String>> trainingSentenceCollection,
							   Collection<List<String>> validationSentenceCollection,
							   Collection<List<String>> trainValidSentenceCollection,
							   Collection<List<String>> testSentenceCollection,
							   boolean verbose) {
		double hubPerplexity = calculatePerplexity(languageModel,
				extractCorrectSentenceList(speechNBestLists));
		double wsjPerplexityTrain = calculatePerplexity(languageModel,
				trainingSentenceCollection);
		double wsjPerplexityValid = calculatePerplexity(languageModel,
				validationSentenceCollection);
		double wsjPerplexityTrainValid = calculatePerplexity(languageModel,
				trainValidSentenceCollection);
		double wsjPerplexityTest = calculatePerplexity(languageModel,
				testSentenceCollection);
		
		System.out.println("Train WSJ Perplexity:  " + wsjPerplexityTrain);
		System.out.println("Valid WSJ Perplexity:  " + wsjPerplexityValid);
		System.out.println("Training and valid combined WSJ Perplexity:  " + wsjPerplexityTrainValid);
		System.out.println("Test WSJ Perplexity:  " + wsjPerplexityTest);
		System.out.println("HUB Perplexity:  " + hubPerplexity);
		System.out.println();
	
		System.out.println("WER Baselines:");
		System.out.println("  Best Path:  "
				+ calculateWordErrorRateLowerBound(speechNBestLists));
		System.out.println("  Worst Path: "
				+ calculateWordErrorRateUpperBound(speechNBestLists));
		System.out.println("  Avg Path:   "
				+ calculateWordErrorRateRandomChoice(speechNBestLists));
		double wordErrorRate = calculateWordErrorRate(languageModel,
				speechNBestLists, verbose);

	    System.out.println("HUB Word Error Rate: " + wordErrorRate);
		System.out.println("Generated Sentences:");
	   for (int i = 0; i < 10; i++)
	   System.out.println("  " + languageModel.generateSentence());
	}

	private static void printArray(double[][] array) {
		System.out.println("        0.1      0.2      0.3      0.4      0.5      0.6      0.7      0.8      0.9      1.0");
		for (int i = 0; i < 10; i ++) {
			System.out.format("%2.1f :  ", i * 0.1 + 0.1);
			for (int j = 0; j < 10; j ++) {
				System.out.format("%08.5f  ", array[i][j]);
			}
			System.out.format("\n");
		}
		System.out.println();
	}

	private static void printArray(double[] array) {
		System.out.println("0.1      0.2      0.3      0.4      0.5      0.6      0.7      0.8      0.9      1.0");
		for (int i = 0; i < 10; i ++) {
			System.out.format("%08.5f  ", array[i]);
		}
		System.out.println();
	}
}
