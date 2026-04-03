# 📤 GitHub Upload Instructions

Follow these steps to upload your 21 Days ML/DL/GenAI project portfolio to GitHub.

---

## 🎯 Prerequisites

1. **Git installed** on your computer
   - Download from: https://git-scm.com/downloads
   - Verify installation: Open Command Prompt and type `git --version`

2. **GitHub account** created
   - Sign up at: https://github.com/signup

---

## 🚀 Step-by-Step Guide

### Step 1: Open Command Prompt/Terminal

Navigate to your project folder:
```bash
cd "c:\Users\Rasak\Desktop\coding\GFG course Project"
```

### Step 2: Initialize Git Repository

```bash
git init
```

This creates a `.git` folder (hidden) that tracks your project changes.

### Step 3: Add All Files

```bash
git add .
```

This stages all files for commit. The `.gitignore` file ensures unnecessary files (like `__pycache__`) are excluded.

### Step 4: Create Initial Commit

```bash
git commit -m "Initial commit: 21 Days ML/DL/GenAI Portfolio - All 21 projects with Streamlit apps"
```

### Step 5: Create GitHub Repository

1. Go to GitHub: https://github.com/new
2. **Repository name**: `21-days-ml-dl-genai` (or your preferred name)
3. **Description**: "Complete portfolio from GeeksforGeeks: 21 Projects covering ML, Deep Learning & GenAI with interactive Streamlit apps"
4. **Visibility**: Choose Public (to showcase) or Private
5. **DO NOT** initialize with README (we already have one)
6. Click **"Create repository"**

### Step 6: Connect Local Repository to GitHub

GitHub will show you commands. Use these (replace YOUR_USERNAME with your GitHub username):

```bash
git remote add origin https://github.com/YOUR_USERNAME/21-days-ml-dl-genai.git
git branch -M main
git push -u origin main
```

**Example:**
```bash
git remote add origin https://github.com/johndoe/21-days-ml-dl-genai.git
git branch -M main
git push -u origin main
```

### Step 7: Enter GitHub Credentials

When prompted:
- **Username**: Your GitHub username
- **Password**: Use a **Personal Access Token** (not your regular password)

#### How to Create Personal Access Token:
1. GitHub → Settings (click your profile picture)
2. Developer settings (bottom left)
3. Personal access tokens → Tokens (classic)
4. Generate new token (classic)
5. Name: "Upload ML Portfolio"
6. Expiration: 90 days (or custom)
7. Select scopes: ✅ **repo** (full control)
8. Generate token
9. **COPY THE TOKEN** (you won't see it again!)
10. Use this token as your password when pushing

### Step 8: Verify Upload

1. Go to your GitHub repository URL
2. You should see all 21 Day folders, README.md, and other files
3. Click on README.md to see your beautiful portfolio page!

---

## 🔄 Future Updates

When you make changes to your projects:

```bash
# 1. Add changed files
git add .

# 2. Commit with a descriptive message
git commit -m "Updated Day 5: Added new clustering visualization"

# 3. Push to GitHub
git push
```

---

## 📁 Repository Structure After Upload

```
21-days-ml-dl-genai/
├── README.md                 ← Main portfolio page
├── requirements.txt          ← All dependencies
├── .gitignore               ← Excluded files
├── Day 1/
│   ├── app_day1.py
│   └── README_DAY1.md
├── Day 2/
│   ├── app_day2.py
│   └── README_DAY2.md
├── ... (Days 3-20)
├── Day 21/
│   ├── app_day21.py
│   └── README_DAY21.md
└── Datasets/
    ├── Titanic-Dataset.csv
    ├── netflix_titles.csv
    └── ... (other datasets)
```

---

## ✨ Customization Tips

### Update README.md with Your Info

1. Open `README.md`
2. Replace `YOUR_USERNAME` with your GitHub username
3. Update the **Contact** section with your name
4. Add your LinkedIn, email, or portfolio links

### Add Screenshots (Optional)

1. Create a `screenshots/` folder
2. Take screenshots of each day's app
3. Add images to README:
   ```markdown
   ![Day 1 - Titanic EDA](screenshots/day1.png)
   ```

### Create a LICENSE File (Optional)

```bash
# In your project folder
echo "MIT License - Add your name and year" > LICENSE
git add LICENSE
git commit -m "Added LICENSE"
git push
```

---

## 🎨 Make Your Repository Stand Out

### Add GitHub Topics

On your repository page:
1. Click ⚙️ Settings (repo settings, not account)
2. Add topics: `machine-learning`, `deep-learning`, `streamlit`, `genai`, `python`, `data-science`, `portfolio`

### Pin Your Repository

1. Go to your GitHub profile
2. Click "Customize your pins"
3. Select this repository
4. It will appear prominently on your profile!

### Add Repository Description

On repository main page:
1. Click the ⚙️ gear icon next to "About"
2. Description: "21 interactive Streamlit apps covering ML, DL & GenAI | GeeksforGeeks Course Portfolio"
3. Website: Your deployed Streamlit Cloud URL (if you deploy)
4. Topics: Add relevant tags

---

## 🚀 Bonus: Deploy to Streamlit Cloud (Free!)

### Make Your Apps Live on the Internet

1. Go to: https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `21-days-ml-dl-genai`
5. Main file path: `Day 1/app_day1.py` (or any day)
6. Click "Deploy!"

Your app will be live at: `https://[app-name].streamlit.app`

**Deploy all 21 apps separately** to showcase your entire portfolio!

---

## ❓ Troubleshooting

### Issue: "Git is not recognized"
**Solution**: Install Git from https://git-scm.com/downloads and restart Command Prompt

### Issue: "Permission denied (publickey)"
**Solution**: Use HTTPS (not SSH) URL when adding remote:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/repo-name.git
```

### Issue: "Updates were rejected"
**Solution**: Pull first, then push:
```bash
git pull origin main --rebase
git push
```

### Issue: Large files rejected
**Solution**: Some datasets might be too large. Either:
- Remove from tracking: `git rm --cached Datasets/large_file.csv`
- Use Git LFS: https://git-lfs.github.com/

---

## 📞 Need Help?

- GitHub Docs: https://docs.github.com/
- Git Tutorial: https://www.atlassian.com/git/tutorials
- Streamlit Community: https://discuss.streamlit.io/

---

## 🎉 Success!

Once uploaded, share your repository:
- LinkedIn post: "Just completed 21 days of ML/DL/GenAI! Check out my portfolio: [GitHub URL]"
- Resume: Add GitHub URL under Projects
- Portfolio website: Link to your repository

**Your work is now publicly showcased!** 🌟

---

**Good luck with your GitHub upload! You've built something amazing!** 🚀
