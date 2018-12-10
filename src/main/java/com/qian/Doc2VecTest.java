package com.qian;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by qianyang on 8/28/2018.
 */
public class Doc2VecTest {
	private static Logger log = LoggerFactory.getLogger(Doc2VecTest.class);
	//文档向量输出路径
	private static String outputPath = "data/raw_sentences.txt";
	private static String inputPath = "data/doc2vec.txt";
	public static void main(String[] args) throws Exception {
		//输入文本文件的目录
		File inputTxt = new File(inputPath);
		log.info("开始加载数据...." + inputTxt.getName());
		//加载数据
		SentenceIterator iter = new LineSentenceIterator(inputTxt);
		//切词操作
		TokenizerFactory token = new DefaultTokenizerFactory();
		//去除特殊符号及大小写转换操作
		token.setTokenPreProcessor(new CommonPreprocessor());
		AbstractCache<VocabWord> cache=new AbstractCache<>();
		//添加文档标签，这个一般从文件读取，为了方面我这里使用了数字
		List<String> labelList = new ArrayList<String>();
		for (int i = 1; i < 299395; i++) {
			labelList.add("doc"+i);
		}
		//设置文档标签
		LabelsSource source = new LabelsSource(labelList);
		log.info("训练模型....");
		ParagraphVectors vec = new ParagraphVectors.Builder()
				.minWordFrequency(1)
				.iterations(5)
				.epochs(1)
				.layerSize(50)
				.learningRate(0.025)
				.labelsSource(source)
				.windowSize(5)
				.iterate(iter)
				.trainWordVectors(false)
				.vocabCache(cache)
				.tokenizerFactory(token)
				.sampling(0)
				.build();

		vec.fit();
		log.info("相似的句子:");
		Collection<String> lst = vec.wordsNearest("doc1", 10);
		System.out.println(lst);
		log.info("输出文档向量....");
		writeDocVectors(vec,outputPath);
		//获取某词对应的向量
		log.info("向量获取:");
		double[] docVector = vec.getWordVector("doc1");
		System.out.println(Arrays.toString(docVector));
	}
	public static void writeDocVectors(ParagraphVectors vectors, String outpath) throws IOException {
		BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(outpath)),"gbk"));
		BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(inputPath)), "gbk")); 
		String line = null;
		int i = 1;
		Map<String, String> keyToDoc = new HashMap<>();
		while ((line = bufferedReader.readLine())!=null) {
			keyToDoc.put("doc" + i, line);
			i++;
		}
		VocabCache<VocabWord> vocabCache = vectors.getVocab();
		for (VocabWord word : vocabCache.vocabWords()) {
			StringBuilder builder = new StringBuilder();
			//获取每个文档对应的标签
			INDArray vector = vectors.getWordVectorMatrix(word.getLabel());
			//向量添加
			for (int j = 0; j < vector.length(); j++) {
				builder.append(vector.getDouble(j));
				if (j < vector.length() - 1) {
					builder.append(" ");
				}
			}
			//写入指定文件
			bufferedWriter.write(keyToDoc.get(word.getLabel()) + "\t" + builder.append("\n").toString());
		}
		bufferedWriter.close();
		bufferedReader.close();
	}
}
