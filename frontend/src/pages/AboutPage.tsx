// frontend/src/pages/AboutPage.tsx
import React from 'react';

const AboutPage: React.FC = () => {
  return (
    // Apply consistent max-width and padding as used in DemoPage's wrapper
    <div className="container mx-auto p-4 max-w-4xl">
      <h1 className="text-3xl font-bold text-center mb-8 text-gray-800">
        About This Recommender
      </h1>

      <div className="bg-white p-6 md:p-8 rounded-lg shadow-lg space-y-6">

        <section>
          <h2 className="text-2xl font-semibold text-indigo-700 mb-3">Project Goal</h2>
          <p className="text-gray-700 leading-relaxed">
            The goal of this project is to build a course recommendation system using the
            Open University Learning Analytics Dataset (OULAD). We aim to recommend relevant
            course presentations (specific offerings of a course module, like 'AAA_2013J')
            to students based on their historical interactions within the Open University's
            Virtual Learning Environment (VLE), demographics, and the characteristics of the
            courses themselves. The idea is to help students discover courses they might find
            engaging or useful based on the patterns learned from past student data.
          </p>
        </section>

        <hr className="my-6"/>

        <section>
          <h2 className="text-2xl font-semibold text-indigo-700 mb-3">The OULAD Dataset</h2>
          <p className="text-gray-700 leading-relaxed">
            The OULAD dataset contains anonymized information about students, their demographics,
            registered courses (modules and presentations), interactions with the VLE (clicks on
            resources, forums, quizzes, etc.), and assessment results across seven courses.
            It's a rich dataset for analyzing student learning behaviors and building predictive models.
            For this recommender, we primarily used the `studentInfo`, `courses`, `studentRegistration`,
            and `studentVle` tables.
          </p>
        </section>

        <hr className="my-6"/>

        <section>
          <h2 className="text-2xl font-semibold text-indigo-700 mb-3">How Recommendations are Generated</h2>
          <p className="text-gray-700 leading-relaxed mb-4">
            This demonstration uses an **Item-Based Collaborative Filtering (ItemCF)** model, which
            performed best among several models evaluated during development. Here's a simplified overview
            of the process:
          </p>
          <ol className="list-decimal list-inside space-y-3 text-gray-700">
            <li>
              <strong>Preprocessing & Implicit Feedback:</strong> Raw VLE clickstream data is processed. We filter out inactive periods and aggregate interactions per student for each course presentation they took. A key step is calculating an "implicit feedback" score, representing how engaged a student was with a presentation. In this project, we used <code className="bg-gray-200 text-sm p-1 rounded">log1p(total_clicks)</code> – meaning the more clicks, the higher the engagement score (using a log scale to temper the effect of extremely high click counts). We also filter out users and items with very few interactions to focus on more reliable data.
            </li>
            <li>
              <strong>Item Similarity Calculation:</strong> The core idea of ItemCF is: "Courses that similar students interact with are similar". The model builds a matrix of students vs. the course presentations they interacted with (using the implicit feedback score). It then calculates the similarity (using cosine similarity) between pairs of course presentations based on which students interacted with them and how much. High similarity means students who took Course A often also took/engaged highly with Course B.
            </li>
            <li>
              <strong>Prediction Generation:</strong> When you select a student ID:
              <ul className="list-disc list-inside ml-6 mt-2 space-y-1">
                <li>The system retrieves the course presentations the selected student previously interacted with (from the training data).</li>
                <li>It looks up the pre-calculated similarity between those "seen" courses and all other "candidate" courses the student *hasn't* seen.</li>
                <li>A score is calculated for each candidate course based on how similar it is to the courses the student liked/interacted with in the past (weighted by the similarity scores and the student's past engagement).</li>
                <li>The candidate courses are ranked by this predicted score.</li>
              </ul>
            </li>
             <li>
              <strong>Top-K Recommendations:</strong> The system displays the Top-K (e.g., Top 10) highest-scoring, previously unseen course presentations as recommendations.
            </li>
          </ol>
        </section>

         <hr className="my-6"/>

         <section>
            <h2 className="text-2xl font-semibold text-indigo-700 mb-3">Evaluation</h2>
             <p className="text-gray-700 leading-relaxed">
                Models were evaluated using a time-based split (training on earlier interactions, testing on later ones)
                to simulate predicting future interests. Key metrics like Precision@10, Recall@10, and NDCG@10 were used.
                ItemCF demonstrated the best performance, particularly in Recall and NDCG, indicating its effectiveness in
                identifying and ranking relevant items for users within this dataset context. (See `reports/final_report.md` for detailed results).
            </p>
         </section>

         <hr className="my-6"/>

        <section>
          <h2 className="text-2xl font-semibold text-indigo-700 mb-3">Limitations</h2>
          <ul className="list-disc list-inside space-y-2 text-gray-700">
            <li>
              <strong>Cold Start:</strong> This model cannot generate recommendations for students with no prior interaction history in the training data, nor can it recommend brand new courses added after the model was trained.
            </li>
            <li>
              <strong>Implicit Feedback Nuances:</strong> Click count is a proxy for engagement; it doesn't capture satisfaction or learning outcomes directly.
            </li>
             <li>
              <strong>Limited Items:</strong> After filtering, the number of unique course presentations in the training data (22) is relatively small, which can limit the diversity of recommendations.
            </li>
             <li>
              <strong>Static Model:</strong> The model currently loaded is based on a snapshot of the data. A production system would need mechanisms for periodic retraining.
            </li>
          </ul>
        </section>

      </div>
    </div>
  );
};

export default AboutPage;