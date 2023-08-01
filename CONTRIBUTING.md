Contributing
============

This is a brief guide to contributing to the LtU-ILI code. 

## Identifying Issues
Action items and current work efforts are organized through [Github Issues](https://github.com/maho3/ltu-ili/issues). Branches and pull requests should be created and organized through public issues. For a Kanban board view of the status of current issues, there is also a [Github project page](https://github.com/users/maho3/projects/1). 

For new contributers, issues tagged with 'good first issue' are a good place to start.

## Declaring new issues.
Anyone can add new issues to the repository's [Github Issues](https://github.com/maho3/ltu-ili/issues) page. When adding a new issue, make sure to give it a descriptive name and comments so others can understand what needs to be done. Also, be sure to use the panels on the right of the New Issue page to associate it with the Kanban board (Projects > LtU Express v0) and to tag it with the correct labels (e.g. pipeline, inference, compression, etc.).

## Making and submitting changes.
### Installing ltu-ili
The first step to contributing to the repository is installing it on your machine. Follow [these instructions](INSTALL.md) for getting your environment set up.

### Opening new branches
Once the code is setup and working, you can start tackling your first issue. Find an issue that you're interested in working on in the [issues page](https://github.com/maho3/ltu-ili/issues). The main branch of ltu-ili is protected, meaning that you cannot directly push changes to it without first submitting a pull request from a non-protected branch. All contributing work must then be done in side branches and merged into main after code review by someone with Direct Access.

If a branch to address this issue is already open, its name should be listed on the right panel of the Issue page under 'Development.' If not, you can automatically add a new branch with an appropriate name by clicking the link under the 'Development' tab. You can also do this manually on your local machine with the command,
```bash
    git checkout -b [name_of_your_new_branch]
    git push origin [name_of_your_new_branch]
```
Please ensure that your branch name clearly identifies which issue it is aiming to address, and that it is linked to the appropriate Issues page. See other branches for naming conventions.

### Editing a branch
Once you've created a new branch for your specific issue, you can push changes to it without affecting the main branch. See [this tutorial](https://docs.github.com/en/get-started/quickstart/hello-world) for getting accustomed to making changes in git branches.

### Making a pull request
Once you are satisfied with your changes and they appropriately address the Issue, you can submit a pull request to submit your code for review. Before initiating a pull request, ensure you have merged the most recent version of the main branch into your new branch.
```bash
    git checkout [name_of_your_new_branch]
    git merge main
    git push
```
In this process, you may run into merge conflicts. [Here's a tutorial](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-using-the-command-line) for dealing with them.

Once you've updated your branch with the newest main changes, you can submit a pull request by navigating to the 'Pull requests' tab in the ltu-ili repository and clicking `New pull request.' Ensure you select the correct branch to merge into main and that you assign a reviewer to approve your merge. Once they have reviewed your code, they will merge your changes into the main branch.

### Documenting changes
Once you've made your changes, you will want to also edit the documentation to reflect these. The [LtU-ILI documentation](https://ltu-ili.readthedocs.io/en/latest/) has been created using Sphinx and is published to ReadTheDocs. To automatically generate documentation, you should format your docstrings to be compatible with Sphinx's `autodoc` extension. [Here's a tutorial](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/index.html) explaining how to use Sphinx and ReadTheDocs.
