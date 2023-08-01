Data sharing
============

This document contains instructions for accessing and sharing data within the Learning the Universe project.

As of 11/21/2021, we have 10 TB of shared data storage on the [Open Storage Network](https://www.openstoragenetwork.org/) (OSN). We can push and pull data from the server through the S3 RESTful API at the following address:

>[https://sdsc.osn.xsede.org/learningtheuniverse](https://sdsc.osn.xsede.org/learningtheuniverse)

**Note**: anyone with this link can download data from the server. For write access, contact Matt Ho (matthew.annam.ho@gmail.com) or any other datamangers.

The official OSN user guide is [here](https://www.openstoragenetwork.org/wp-content/uploads/2021/04/OSN-UserGuide.pdf). However, it's a bit sparse and poorly updated, so below we've highlighted the important ways to access the data.

## Browser interface
Anyone with a CILogon account (home institution, Github, ORCID, etc) can download data from the S3 web portal. 
1. Navigate to [https://portal.osn.xsede.org/s3browser/](https://portal.osn.xsede.org/s3browser/) and sign in with your CILogon account.
2. Click the 'Settings' gear icon in the top-right.
3. Under 'Endpoint', enter https://sdsc.osn.xsede.org. Under 'Bucket', enter learningtheuniverse. All other options can be left as default (Public Bucket, Folder). Click 'Query S3'.
4. You should now see the contents of the bucket. Feel free to explore the available data, and click on a file to download it.


## Command-line interface: 
You can also access the data through the command line using [rclone](https://rclone.org/). Rclone is a command-line utility installed on most HPC systems, and it has similar functionality to scp and rsync (but slightly different syntax!).

To use this, ensure rclone is installed on your local or HPC machine. Then, use the command `rclone config file` to find the location of the configuration file. Open this file in a text editor and add the following lines to the end:

```
[ltu]
type = s3
provider = Ceph
access_key_id = QFRAS2I9QFRMM33E0X35
secret_access_key = QrOQRhBHPkL9HT3p9AP7QU+f3tPFrs
endpoint = https://sdsc.osn.xsede.org
```
This will configure your local rclone to access the LtU OSN server. You can then interact with the server using [rclone commands](https://rclone.org/docs/). For example, to list all files available on the remote, you can use:
```bash
rclone ls ltu:/learningtheuniverse/
```
**Warning**: Some rclone commands will work differently than other things you are familiar with (such as `scp` or `rsync`). Be sure to check the documentation before using a new command.

## Globus
OSN has a globus endpoint named “osn s3 collection”, but I have had authentication issues with it and haven’t gotten it to work yet.


## Questions?
Contact Matt Ho (by Slack or by email, matthew.annam.ho@gmail.com).