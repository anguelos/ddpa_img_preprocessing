I want you to read and write in you directory, employ git only passively, run pytest and sphix.
Consider your self restricted, I am going to be manually managing git, and manually running any pip commands, including setup.py install ...

This directory already contains a very brief outline of the project.
Read all of these instructions before implementing anything.
Ask me step-by-step questions on any clarifications you might need.
If you see an instruction that seems ambiguous or suggests a poor design pattern / choice. Suggest alternatives but be brief.


FSDB architecture:
This project will implement an application that sits on top of a database implemented on the Filesystem called FSDB.
The database provides acces to medieval charters. 
The database can ocasionaly be corrupted.
The database has the following hierarchy where each directory is an entity in a table and attributes are files: root->Archive->Fond->Charter.
An example FSDB containing a single charter can be seen in test/fake_fsdb_root/.
Charter directories contain one or more images of the object. These images follow the naming convention img_md5_sum.img.imgfile_ext and are considered to be immutable.
Archives have strictly alphanumeric names while fonds and charters have md5 hashes for names. Each charter directory contains the following:
1)CH.cei.xml: an xml document containg transcriptions, summaries, and metadata of the charter in the dir.
2)CH.url.txt: a txt file the URL the website from which this data was leeched.
3)CH.atom_id.txt: a txt file containing an unique id of the charter from another domain which is passed to generate the charters md5sum name
4)image_urls.json: which associates the image filename in FSDB with the URL from which it was obtained. Note in some cases the image_urls.json refers to images as img_md5_sum.imgfile_ext instead of img_md5_sum.img.imgfile_ext which is what is on the filesystem.
5)An optional json file that was computed from a YOLO object detector might be available for every image file with the name img_md5_sum.layout.pred.json, manually annotated files can also be available under the name img_md5_sum.layout.gt.json . These files will be refered to as layout files.

Your modus operanti:
I want you to maintain .claude/DESIGN.md where you will be storing design patterns favored in the project, as well a conventions, decisions etc... .claude/DESIGN.md should be able to work as a prompt that will as much as possible function like a resume session.
Any instuction requires you to run a command you are not authorized offer me the option to add to your list of available commands.
If not sure ask me quick step by step questions.
This project should build heavilly on my pytorch_mentor project. You can assume it's code will reside in ../torch_mentor. You should have the right to read anything in it and with my explicit per file permission modify / create files in it.
Accessing files/pages on the internet should only be done with my explicit permission.
This file (.claude/CLAUDE_INITIAL.md) should stay immutable.
Whenever an instruction overides this text update file (.claude/CLAUDE_MODIFIED.md) ideally a meld or a diff should be informative of the evolution of the project, both you and me can use .claude/CLAUDE_MODIFIED.md to register the present state or require updates to the project.


Python coding:
For this project I want you to become familiar with my pytorch training framework torcvh_mentor (https://github.com/anguelos/torch_mentor) and fargv (https://github.com/anguelos/fargv) for argument parsing.
Fargv is slightly unorthodox ask me how to use it if not clear specifically positionals, choices and default booleans.
Entry point functions should always start with def main_
Docstrings must be numpy strings contain Inputs, Return Values, Raised Exceptions, code snippets for examples etc. In general when prototyping dont implement the automatically, avoid implemeting them when scafolding, Implement them when explicitly asked for it.
As with docstrings, when scaffolding code you should avoid extencive typehints. But when asked to make code more formal or to produce docstrings, add typehints beeing as specific as possible. Use python's standard typing module and its type constructs such as Optional,List,Tuple,Dict, generator etc...
Many functions that will have a heavy workload should have a verbose integer or boolean parameter.
For heavy outer loops tqdm progress bars should be employed but only enabled if verbose was passed, as always ommit the.
When dealing image processing functions parameters should be prefferable Pillow images or strings/paths which would be implicitly loaded by the outer most function.


Python project structure:
I want you to create a setup.py that will be the main tool realising the deployment instalation etc. In the setup.py be as extencive in requirements.
Create a minimum pyproject.toml which will delegate as much as possible to setup.py and contain linting information for ruff, the allowed linewidth will be 160 characters.
Add a requirements.txt that will be enough to install a proper development python environment.
Create and maintain a README.md
Set the project license to the Afero public licence.
Add a test directory where every kind of test case will have its onw directory among which specifcally a test/unittest subdirectory. The test cases should be implemented with pytest. Unittests should be used for coverage and they should maximize coverage of ./src/ but exclude all entry points (def main_....) when computing coverage.
Add a docs subdirectory which will use sphinx to render documentation. The documentation will be using markdown instead rst files. It will be offering an API description, command line tools documentation, a quickstart guide etc... The documentation  will eventually be published in RTD so prefer its style. Employ popular features such as copying code and linking to source code.
create a Makefile that will allow to: 
    1)"make clean": erase all build files as well as .pyc files, .egg-inf etc..
    2)"make build": builds the file for deployment to pypi (erases out of data files in )
    3)"make doc", "make htmldoc": build sphinx for html output
    4)"make pdfdoc": build sphinx for single pdf output
    5)"make test": runs all testcases but exits on the first failure.
    6)"make testfull" runs all testcases but doesnt exit on the first failure.
    7)"make unitest" runs all only unitests and print coverage


Project outline:
This project aims at performing preprocessing of the images for deeper analysis follwing the processing pipelines.
When running in the mode called "offline computation" the script should be computing fundamentally three things:
1)Image binarization
2)Image rescaling to an aproximated uniform real world resolution (estimated PPI)
3)Choosing the most apropriate image for looking at the charters textual and non-textual content (recto/verso/other)
Initially for scafolding testing etc. these tasks should be implemted using resonable heuristics.
In later development these tasks will be implemented by pytorch models using my mentor framework.
Naive and ML implemetations should be implemented as functors sharing a common interface.
Utilities for training and testing ML and heuristics should exist as main entry level scripts.


Specifically:

1)Image binarization: This module (ddp_binarize) should implement functors (heuristic and later not) that take an image as an input and return an image of the same resolution that is semantically binarized to text non-text.
In the module there is a dibco.py file implementing a uniform wrapper around all the dibco datasets, as many of the files are not available I have cached a directory which I made publicly available in google drive: https://drive.google.com/drive/folders/1IYuPOPiGsrFAf-dfK4mZCwrB9v37WLv3 extend dibco.py to download with gdown this cache if any of the URLs is broken. The script should take inputs using a fargv set of image paths and produce for image path named ..../img_md5_sum.img.imgfile_ext a file ...../img_md5_sum.bin.png. The ouputs must not be literary binary but grayscales with 0 meaning certainly foreground, 255 meaning certainly background, any value smaller than 128 should mean probably foreground and any value greater than 128 should mean probably backgorund.

2)Image Resolution: This module (ddp_resolution) should implement functors (heuristic and later on pytorch) that given an image will return an estimate of PixelsPerInch. The standalone inference script should take charter images as inputs and if possible save  a json file with the estimated PPI and confidence, optionally the script should allow to save a resized vesion of the image to a desired PPI passed by fargv along with a max size and a min size in the name from img_md5_sum.img.imgfile_ext to img_md5_sum.scaled_ppi_N.png .

3)Find Recto: This module (ddp_recto) is tasked with finding the most apropriate image to extract and analyse the textual content. Among two images containing the tighter crop and a broader crop, the tighter crop is less apropriate. Essentially this functor should take as an argumend a charter directory and return a list of tuples with image paths (relative to the charter dir) and probabillity it is the most apropriate. There are cases where although there are one or more images, none is apropriate (maybe they are thumbnails, maybe they are bookscans of summaries), in that case the highest probabillity should be less than 0.5 . The standalone script should offer the option to create a symbolic link to  the most probable image if prob above 0.5 which would have the name CH.img.imgfile_ext . The heuristic functor should prefer the file occuring first in image_urls.json if the respective img_md5_sum.seals.pred.json has relatively large "Img:WritableArea" and mostly inside it a single "Wr:OldText" that dominates the surface.

4)Putting it all together: This module (ddp_cv_preprocess) should host any common code specifically to the Handling the FSDB etc. It should also have a main inference script called "ddp_cv_preprocess_offline" that will be computing the preprocessing information.


How I want you to proceed:
1)Make shure the local settings .json in .claude allows you to implement all these instructions.
2)Create scaffold code for all modules and entry points
3)Create the project files I mentioned earlier.
3)Create simple unittests providing minimal coverage to the modules.
4)Create sphinx documentation
5)Update .claude/CLAUDE_MODIFIED.md
