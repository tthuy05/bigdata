const { d, FONT } = require("./lib");
const { Paragraph, TextRun, AlignmentType } = d;
const REFS = [
  "K. Shvachko, H. Kuang, S. Radia, and R. Chansler, \"The Hadoop Distributed File System,\" in Proc. IEEE 26th Symp. Mass Storage Systems and Technologies (MSST), 2010, pp. 1–10.",
  "V. K. Vavilapalli et al., \"Apache Hadoop YARN: Yet Another Resource Negotiator,\" in Proc. 4th ACM Symp. Cloud Computing (SoCC), 2013.",
  "J. Dean and S. Ghemawat, \"MapReduce: Simplified Data Processing on Large Clusters,\" Communications of the ACM, vol. 51, no. 1, pp. 107–113, 2008.",
  "M. Armbrust et al., \"Spark SQL: Relational Data Processing in Spark,\" in Proc. ACM SIGMOD Int. Conf. Management of Data, 2015, pp. 1383–1394.",
  "X. Meng et al., \"MLlib: Machine Learning in Apache Spark,\" Journal of Machine Learning Research, vol. 17, no. 34, pp. 1–7, 2016.",
  "Apache Software Foundation, \"ML Pipelines – Spark 3.5 Documentation.\" [Online]. Available: https://spark.apache.org/docs/latest/ml-pipeline.html",
  "Apache Software Foundation, \"Classification and Regression – Spark MLlib Documentation.\" [Online]. Available: https://spark.apache.org/docs/latest/ml-classification-regression.html",
  "M. Zaharia et al., \"Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing,\" in Proc. 9th USENIX Symp. Networked Systems Design and Implementation (NSDI), 2012.",
  "M. Zaharia et al., \"Apache Spark: A Unified Engine for Big Data Processing,\" Communications of the ACM, vol. 59, no. 11, pp. 56–65, 2016.",
  "E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, \"PaySim: A Financial Mobile Money Simulator for Fraud Detection,\" in Proc. 28th European Modeling and Simulation Symposium (EMSS), Larnaca, Cyprus, 2016.",
  "E. Lopez-Rojas, \"Synthetic Financial Datasets For Fraud Detection,\" Kaggle. [Online]. Available: https://www.kaggle.com/datasets/ealaxi/paysim1",
  "L. Breiman, \"Random Forests,\" Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
  "J. H. Friedman, \"Greedy Function Approximation: A Gradient Boosting Machine,\" The Annals of Statistics, vol. 29, no. 5, pp. 1189–1232, 2001.",
  "L. Breiman, J. Friedman, R. Olshen, and C. Stone, Classification and Regression Trees. Wadsworth, 1984.",
  "R. J. Bolton and D. J. Hand, \"Statistical Fraud Detection: A Review,\" Statistical Science, vol. 17, no. 3, pp. 235–255, 2002.",
  "A. Abdallah, M. A. Maarof, and A. Zainal, \"Fraud Detection System: A Survey,\" Journal of Network and Computer Applications, vol. 68, pp. 90–113, 2016.",
  "H. He and E. A. Garcia, \"Learning from Imbalanced Data,\" IEEE Trans. Knowledge and Data Engineering, vol. 21, no. 9, pp. 1263–1284, 2009.",
  "A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, and G. Bontempi, \"Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy,\" IEEE Trans. Neural Networks and Learning Systems, vol. 29, no. 8, pp. 3784–3797, 2018.",
  "F. Carcillo, A. Dal Pozzolo, Y.-A. Le Borgne, O. Caelen, Y. Mazzer, and G. Bontempi, \"SCARFF: A Scalable Framework for Streaming Credit Card Fraud Detection with Spark,\" Information Fusion, vol. 41, pp. 182–194, 2018.",
  "T. Saito and M. Rehmsmeier, \"The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets,\" PLoS ONE, vol. 10, no. 3, e0118432, 2015.",
  "T. Chen and C. Guestrin, \"XGBoost: A Scalable Tree Boosting System,\" in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 785–794.",
  "D. W. Hosmer, S. Lemeshow, and R. X. Sturdivant, Applied Logistic Regression, 3rd ed. Wiley, 2013.",
  "C. Phua, V. Lee, K. Smith, and R. Gayler, \"A Comprehensive Survey of Data Mining-based Fraud Detection Research,\" arXiv:1009.6119, 2010.",
  "D. Laney, \"3D Data Management: Controlling Data Volume, Velocity, and Variety,\" META Group Research Note, 2001.",
  "M. Armbrust et al., \"Structured Streaming: A Declarative API for Real-Time Applications in Apache Spark,\" in Proc. ACM SIGMOD Int. Conf. Management of Data, 2018.",
  "S. M. Lundberg and S.-I. Lee, \"A Unified Approach to Interpreting Model Predictions,\" in Advances in Neural Information Processing Systems 30 (NeurIPS), 2017.",
  "J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy, and A. Bouchachia, \"A Survey on Concept Drift Adaptation,\" ACM Computing Surveys, vol. 46, no. 4, 2014.",
];
const references = [
  new Paragraph({ spacing: { before: 200, after: 100 }, keepNext: true,
    children: [new TextRun({ text: "TÀI LIỆU THAM KHẢO", bold: true, font: FONT, size: 23 })] }),
  ...REFS.map((r, i) => new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 40, line: 230 },
    indent: { left: 400, hanging: 400 },
    children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 17 }), new TextRun({ text: r, font: FONT, size: 17 })],
  })),
];


module.exports = { references };
