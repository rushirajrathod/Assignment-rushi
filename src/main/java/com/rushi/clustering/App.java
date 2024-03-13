package com.rushi.clustering;

import java.io.File;

import net.sf.javaml.clustering.evaluation.AICScore;
import net.sf.javaml.clustering.evaluation.BICScore;
import net.sf.javaml.clustering.evaluation.ClusterEvaluation;
import net.sf.javaml.clustering.evaluation.SumOfAveragePairwiseSimilarities;
import net.sf.javaml.clustering.evaluation.SumOfSquaredErrors;

import net.sf.javaml.clustering.AQBC;
import net.sf.javaml.clustering.Clusterer;
import net.sf.javaml.clustering.KMeans;
import net.sf.javaml.clustering.FarthestFirst;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.tools.data.FileHandler;

public class App {
    public static void main(String[] args) throws Exception {
        // Load the dataset
        Dataset data = FileHandler.loadDataset(new File("datasets/iris.data"), 4, ",");

        // Define clustering algorithms
        Clusterer km = new KMeans();
        Clusterer far = new FarthestFirst();
        Clusterer adapt = new AQBC();

        // Perform clustering and measure execution time
        long startTime = System.nanoTime();
        Dataset[] clusters1 = km.cluster(data);
        long endTime = System.nanoTime();
        long durationKMEAN = endTime - startTime;

        startTime = System.nanoTime();
        Dataset[] clusters2 = adapt.cluster(data);
        endTime = System.nanoTime();
        long durationAQBC = endTime - startTime;

        startTime = System.nanoTime();
        Dataset[] clusters3 = far.cluster(data); // Changed the clustering algorithm
        endTime = System.nanoTime();
        long durationFarthestFirst = endTime - startTime; // Changed variable name

        // Evaluate clustering results
        ClusterEvaluation aic = new AICScore();
        ClusterEvaluation bic = new BICScore();
        ClusterEvaluation sse = new SumOfSquaredErrors();
        ClusterEvaluation saps = new SumOfAveragePairwiseSimilarities();

        double aicScore3 = aic.score(clusters1);
        double bicScore3 = bic.score(clusters1);
        double sseScore3 = sse.score(clusters1);
        double sapScore3 = saps.score(clusters1);

        double aicScore4 = aic.score(clusters2);
        double bicScore4 = bic.score(clusters2);
        double sseScore4 = sse.score(clusters2);
        double sapScore4 = saps.score(clusters2);

        double aicScore5 = aic.score(clusters3);
        double bicScore5 = bic.score(clusters3);
        double sseScore5 = sse.score(clusters3);
        double sapScore5 = saps.score(clusters3);

        // Print results
        System.out.println(
            "\n*******************************************************************************************");
        for(int i = 0; i < clusters1.length;i++ ){
            System.out.println(clusters1[i]);
        }
        System.out.println("\n");
        for(int i = 0; i < clusters2.length;i++ ){
            System.out.println(clusters2[i]);
        }
        System.out.println("\n");
        for(int i = 0; i < clusters3.length;i++ ){
            System.out.println(clusters3[i]);
        }
            
        System.out.println(
                "\n*******************************************************************************************");
        System.out.println("KMeans Cluster count: " + clusters1.length);
        System.out.println("AQBC Cluster count: " + clusters2.length);
        System.out.println("FarthestFirst Cluster count: " + clusters3.length + "\n");
        System.out.println("\t\t\t KMEAN \t\t\tAQBC \t\t\tFarthestFirst "); // Update the algorithm name
        System.out.println("AIC score:\t\t " + aicScore3 + "\t" + aicScore4 + "\t" + aicScore5);
        System.out.println("BIC score:\t\t " + bicScore3 + "\t" + bicScore4 + "\t" + bicScore5);
        System.out.println("Pairwise Similarities:\t " + sapScore3 + "\t" + sapScore4 + "\t" + sapScore5);
        System.out.println("Sum of squared errors:   " + sseScore3 + "\t" + sseScore4 + "\t" + sseScore5);
        System.out.println(
                "Execution time in MS:    " + durationKMEAN / 1000000 + "ms \t\t\t" + durationAQBC / 100000
                        + "ms \t\t\t"
                        + durationFarthestFirst / 100000 + "ms");
        System.out.println(
                "********************************************************************************************\n");

    }
}
