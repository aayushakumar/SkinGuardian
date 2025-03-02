


# 🛡️ SkinGuardian++  
### A Fair, Explainable, Private, and Robust AI System for Early Skin Cancer Detection  

## 📌 Project Overview  
**SkinGuardian++** is an AI-powered desktop application for **early skin cancer detection** with a strong emphasis on **Socially Responsible AI (SRAI)**. It ensures:  
✅ **Fairness** - Bias mitigation across diverse skin tones.  
✅ **Explainability** - Transparent AI decisions using SHAP, Grad-CAM, and LIME.  
✅ **Privacy** - On-device inference using **Federated Learning** to protect sensitive data.  
✅ **Robustness** - Defense against adversarial attacks and real-world noise.  

---
## 🛠️ Setup Instructions  

### 1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/aayushakumar/SkinGuardian.git
cd SkinGuardian
```

### 2️⃣ **Create and Activate a Virtual Environment**  
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

### 4️⃣ **Run the Application**  
```bash
python app.py
```
---

## 🌿 **Branching Workflow**  

We follow a structured **Git branching strategy**:  
```
- main  (Stable version - NO changes required here)
  ├── dev  (Development branch - All features merge here!)
      ├── fairness-module  (Bias mitigation)
      ├── explainability-module  (SHAP, Grad-CAM)
      ├── robustness-module  (Adversarial attacks)
```

### **🛠️ How to Work on a Feature**  
1. **Checkout `dev` branch**  
   ```bash
   git checkout dev
   git pull origin dev  # Get the latest updates
   ```
2. **Go to your feature branch**  
   ```bash
   git checkout feature-name
   ```
 - **(feature-name:fairness-module, explainability-module, robustness-module)**

3. **Make changes, then commit**  
   ```bash
   git add .
   git commit -m "Added xyz feature module"
   ```
4. **Push your changes to GitHub**  
   ```bash
   git push origin feature-name
   ```

---

## 🔁 **Pull Request (PR) Guidelines**  

**All feature branches must be merged into `dev` via a Pull Request (PR).**  

### **Creating a Pull Request (PR)**
1. Push your branch to GitHub:  
   ```bash
   git push origin feature-name
   ```
2. Go to the **GitHub repo** → **Pull Requests** → **New Pull Request**.  
3. Select **`base branch: dev`** and **`compare branch: feature-name`**.  
4. Add a **descriptive PR title** and **detailed description**.  
5. Submit the PR and request a review.  
6. **Once approved, merge the PR into `dev`**.  

---

## 🔐 **Branch Protection Rules**
To ensure stability:  
🔹 **All team members must merge their feature branches into `dev` via PRs**.  
🔹 **Merging to `main` requires an approved PR from `dev`**.  

---

## 📋 **Git Commands Reference**
💡 Commonly used Git commands:

| Action | Command |
|---------|---------|
| Clone the repo | `git clone <repo_url>` |
| Create a new branch | `git checkout -b branch-name` |
| Switch branches | `git checkout branch-name` |
| Update local branch | `git pull origin branch-name` |
| Stage changes | `git add .` |
| Commit changes | `git commit -m "Your message"` |
| Push to remote | `git push origin branch-name` |
| Fetch latest changes | `git fetch origin` |
| Merge changes | `git merge branch-name` |
| Delete a branch (local) | `git branch -d branch-name` |
| Delete a branch (remote) | `git push origin --delete branch-name` |

---

## 📢 **Need Help?**
If you have any issues, reach out to the team or create an **Issue** in the GitHub repository.  
