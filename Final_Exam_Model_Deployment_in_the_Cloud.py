{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Ethy17/Proj/blob/main/Final_Exam_Model_Deployment_in_the_Cloud.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 894
        },
        "id": "o8BsLfDRE-Jr",
        "outputId": "7eb369e7-906a-40b6-bc57-24f7869909bc"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "InternalHashError",
          "evalue": "module '__main__' has no attribute '__file__'\n\nWhile caching the body of `load_model()`, Streamlit encountered an\nobject of type `builtins.function`, which it does not know how to hash.\n\n**In this specific case, it's very likely you found a Streamlit bug so please\n[file a bug report here.]\n(https://github.com/streamlit/streamlit/issues/new/choose)**\n\nIn the meantime, you can try bypassing this error by registering a custom\nhash function via the `hash_funcs` keyword in @st.cache(). For example:\n\n```\n@st.cache(hash_funcs={builtins.function: my_hash_func})\ndef my_func(...):\n    ...\n```\n\nIf you don't know where the object of type `builtins.function` is coming\nfrom, try looking at the hash chain below for an object that you do recognize,\nthen pass that to `hash_funcs` instead:\n\n```\nObject of type builtins.function: <function load_model at 0x7fe01cd19510>\n```\n\nPlease see the `hash_funcs` [documentation](https://docs.streamlit.io/library/advanced-features/caching#the-hash_funcs-parameter)\nfor more details.\n            ",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/streamlit/runtime/legacy_caching/hashing.py\u001b[0m in \u001b[0;36mto_bytes\u001b[0;34m(self, obj, context)\u001b[0m\n\u001b[1;32m    359\u001b[0m             \u001b[0;31m# Hash the input\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 360\u001b[0;31m             \u001b[0mb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34mb\"%s:%s\"\u001b[0m \u001b[0;34m%\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mtname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_to_bytes\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mobj\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcontext\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    361\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/streamlit/runtime/legacy_caching/hashing.py\u001b[0m in \u001b[0;36m_to_bytes\u001b[0;34m(self, obj, context)\u001b[0m\n\u001b[1;32m    622\u001b[0m             \u001b[0;32massert\u001b[0m \u001b[0mcode\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 623\u001b[0;31m             \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_file_should_be_hashed\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcode\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mco_filename\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    624\u001b[0m                 \u001b[0mcontext\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_get_context\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mobj\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/streamlit/runtime/legacy_caching/hashing.py\u001b[0m in \u001b[0;36m_file_should_be_hashed\u001b[0;34m(self, filename)\u001b[0m\n\u001b[1;32m    401\u001b[0m         return file_util.file_is_in_folder_glob(\n\u001b[0;32m--> 402\u001b[0;31m             \u001b[0mfilepath\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_get_main_script_directory\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    403\u001b[0m         ) or file_util.file_in_pythonpath(filepath)\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/streamlit/runtime/legacy_caching/hashing.py\u001b[0m in \u001b[0;36m_get_main_script_directory\u001b[0;34m()\u001b[0m\n\u001b[1;32m    709\u001b[0m         \u001b[0;31m# script path in ScriptRunner.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 710\u001b[0;31m         \u001b[0mabs_main_path\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpathlib\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mPath\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0m__main__\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__file__\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mresolve\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    711\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mstr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mabs_main_path\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparent\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mAttributeError\u001b[0m: module '__main__' has no attribute '__file__'",
            "\nDuring handling of the above exception, another exception occurred:\n",
            "\u001b[0;31mInternalHashError\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-3-dd979d8a4acf>\u001b[0m in \u001b[0;36m<cell line: 11>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      9\u001b[0m   \u001b[0mmodel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mkeras\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmodels\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mload_model\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'plant_classifier.hdf5'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     10\u001b[0m   \u001b[0;32mreturn\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 11\u001b[0;31m \u001b[0mmodel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mload_model\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     12\u001b[0m st.write(\"\"\"\n\u001b[1;32m     13\u001b[0m \u001b[0;31m# Plant Leaf Detection System\"\"\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/streamlit/runtime/legacy_caching/caching.py\u001b[0m in \u001b[0;36mwrapped_func\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    654\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mshow_spinner\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    655\u001b[0m             \u001b[0;32mwith\u001b[0m \u001b[0mspinner\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmessage\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0m_cache\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 656\u001b[0;31m                 \u001b[0;32mreturn\u001b[0m \u001b[0mget_or_create_cached_value\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    657\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    658\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0mget_or_create_cached_value\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/streamlit/runtime/legacy_caching/caching.py\u001b[0m in \u001b[0;36mget_or_create_cached_value\u001b[0;34m()\u001b[0m\n\u001b[1;32m    580\u001b[0m                 \u001b[0;31m# If we generated the key earlier we would only hash those\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    581\u001b[0m                 \u001b[0;31m# globals by name, and miss changes in their code or value.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 582\u001b[0;31m                 \u001b[0mcache_key\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_hash_func\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnon_optional_func\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mhash_funcs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    583\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    584\u001b[0m             \u001b[0;31m# First, get the cache that's attached to this function.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/streamlit/runtime/legacy_caching/caching.py\u001b[0m in \u001b[0;36m_hash_func\u001b[0;34m(func, hash_funcs)\u001b[0m\n\u001b[1;32m    706\u001b[0m     \u001b[0;31m# because this step will be hashing any objects referenced in the function\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    707\u001b[0m     \u001b[0;31m# body.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 708\u001b[0;31m     update_hash(\n\u001b[0m\u001b[1;32m    709\u001b[0m         \u001b[0mfunc\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    710\u001b[0m         \u001b[0mhasher\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mfunc_hasher\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/streamlit/runtime/legacy_caching/hashing.py\u001b[0m in \u001b[0;36mupdate_hash\u001b[0;34m(val, hasher, hash_reason, hash_source, context, hash_funcs)\u001b[0m\n\u001b[1;32m    106\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    107\u001b[0m     \u001b[0mch\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_CodeHasher\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mhash_funcs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 108\u001b[0;31m     \u001b[0mch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mhasher\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mval\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcontext\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    109\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    110\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/streamlit/runtime/legacy_caching/hashing.py\u001b[0m in \u001b[0;36mupdate\u001b[0;34m(self, hasher, obj, context)\u001b[0m\n\u001b[1;32m    383\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mhasher\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mobj\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mAny\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcontext\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mContext\u001b[0m \u001b[0;34m|\u001b[0m \u001b[0;32mNone\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m->\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    384\u001b[0m         \u001b[0;34m\"\"\"Update the provided hasher with the hash of an object.\"\"\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 385\u001b[0;31m         \u001b[0mb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mto_bytes\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mobj\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcontext\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    386\u001b[0m         \u001b[0mhasher\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mb\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    387\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/streamlit/runtime/legacy_caching/hashing.py\u001b[0m in \u001b[0;36mto_bytes\u001b[0;34m(self, obj, context)\u001b[0m\n\u001b[1;32m    372\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    373\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mex\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 374\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mInternalHashError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mex\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mobj\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    375\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    376\u001b[0m         \u001b[0;32mfinally\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/streamlit/runtime/legacy_caching/hashing.py\u001b[0m in \u001b[0;36mto_bytes\u001b[0;34m(self, obj, context)\u001b[0m\n\u001b[1;32m    358\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    359\u001b[0m             \u001b[0;31m# Hash the input\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 360\u001b[0;31m             \u001b[0mb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34mb\"%s:%s\"\u001b[0m \u001b[0;34m%\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mtname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_to_bytes\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mobj\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcontext\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    361\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    362\u001b[0m             \u001b[0;31m# Hmmm... It's possible that the size calculation is wrong. When we\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/streamlit/runtime/legacy_caching/hashing.py\u001b[0m in \u001b[0;36m_to_bytes\u001b[0;34m(self, obj, context)\u001b[0m\n\u001b[1;32m    621\u001b[0m             \u001b[0mcode\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgetattr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mobj\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"__code__\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    622\u001b[0m             \u001b[0;32massert\u001b[0m \u001b[0mcode\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 623\u001b[0;31m             \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_file_should_be_hashed\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcode\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mco_filename\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    624\u001b[0m                 \u001b[0mcontext\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_get_context\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mobj\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    625\u001b[0m                 \u001b[0mdefaults\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgetattr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mobj\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"__defaults__\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/streamlit/runtime/legacy_caching/hashing.py\u001b[0m in \u001b[0;36m_file_should_be_hashed\u001b[0;34m(self, filename)\u001b[0m\n\u001b[1;32m    400\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    401\u001b[0m         return file_util.file_is_in_folder_glob(\n\u001b[0;32m--> 402\u001b[0;31m             \u001b[0mfilepath\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_get_main_script_directory\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    403\u001b[0m         ) or file_util.file_in_pythonpath(filepath)\n\u001b[1;32m    404\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/streamlit/runtime/legacy_caching/hashing.py\u001b[0m in \u001b[0;36m_get_main_script_directory\u001b[0;34m()\u001b[0m\n\u001b[1;32m    708\u001b[0m         \u001b[0;31m# This works because we set __main__.__file__ to the\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    709\u001b[0m         \u001b[0;31m# script path in ScriptRunner.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 710\u001b[0;31m         \u001b[0mabs_main_path\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpathlib\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mPath\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0m__main__\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__file__\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mresolve\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    711\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mstr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mabs_main_path\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparent\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    712\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mInternalHashError\u001b[0m: module '__main__' has no attribute '__file__'\n\nWhile caching the body of `load_model()`, Streamlit encountered an\nobject of type `builtins.function`, which it does not know how to hash.\n\n**In this specific case, it's very likely you found a Streamlit bug so please\n[file a bug report here.]\n(https://github.com/streamlit/streamlit/issues/new/choose)**\n\nIn the meantime, you can try bypassing this error by registering a custom\nhash function via the `hash_funcs` keyword in @st.cache(). For example:\n\n```\n@st.cache(hash_funcs={builtins.function: my_hash_func})\ndef my_func(...):\n    ...\n```\n\nIf you don't know where the object of type `builtins.function` is coming\nfrom, try looking at the hash chain below for an object that you do recognize,\nthen pass that to `hash_funcs` instead:\n\n```\nObject of type builtins.function: <function load_model at 0x7fe01cd19510>\n```\n\nPlease see the `hash_funcs` [documentation](https://docs.streamlit.io/library/advanced-features/caching#the-hash_funcs-parameter)\nfor more details.\n            "
          ]
        }
      ],
      "source": [
        "import streamlit as st\n",
        "import tensorflow as tf\n",
        "import cv2\n",
        "from PIL import Image,ImageOps\n",
        "import numpy as np\n",
        "\n",
        "@st.cache(allow_output_mutation=True)\n",
        "def load_model():\n",
        "  model=tf.keras.models.load_model('plant_classifier.hdf5')\n",
        "  return model\n",
        "model=load_model()\n",
        "st.write(\"\"\"\n",
        "# Plant Leaf Detection System\"\"\"\n",
        ")\n",
        "file=st.file_uploader(\"Choose plant photo from computer\",type=[\"jpg\",\"png\"])\n",
        "\n",
        "def import_and_predict(image_data,model):\n",
        "    size=(64,64)\n",
        "    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)\n",
        "    img=np.asarray(image)\n",
        "    img_reshape=img[np.newaxis,...]\n",
        "    prediction=model.predict(img_reshape)\n",
        "    return prediction\n",
        "if file is None:\n",
        "    st.text(\"Please upload an image file\")\n",
        "else:\n",
        "    image=Image.open(file)\n",
        "    st.image(image,use_column_width=True)\n",
        "    prediction=import_and_predict(image,model)\n",
        "    class_names=['Alpinia Galanga (Rasna)','Amaranthus Viridis (Arive-Dantu)',\n",
        "                 'Artocarpus Heterophyllus (Jackfruit)',\n",
        "                 'Azadirachta Indica (Neem)','Basella Alba (Basale)',\n",
        "                 'Brassica Juncea (Indian Mustard)','Carissa Carandas (Karanda)',\n",
        "                 'Citrus Limon (Lemon)','Ficus Auriculata (Roxburgh fig)',\n",
        "                 'Ficus Religiosa (Peepal Tree)','Hibiscus Rosa-sinensis',\n",
        "                 'Jasminum (Jasmine)','Mangifera Indica (Mango)',\n",
        "                 'Mentha (Mint)','Moringa Oleifera (Drumstick)',\n",
        "                 'Muntingia Calabura (Jamaica Cherry-Gasagase)',\n",
        "                 'Murraya Koenigii (Curry)', 'Nerium Oleander (Oleander)',\n",
        "                 'Nyctanthes Arbor-tristis (Parijata)','Ocimum Tenuiflorum (Tulsi)',\n",
        "                 'Piper Betle (Betel)','Plectranthus Amboinicus (Mexican Mint)',\n",
        "                 'Pongamia Pinnata (Indian Beech)','Psidium Guajava (Guava)',\n",
        "                 'Punica Granatum (Pomegranate)','Santalum Album (Sandalwood)',\n",
        "                 'Syzygium Cumini (Jamun)','Syzygium Jambos (Rose Apple)',\n",
        "                 'Tabernaemontana Divaricata (Crape Jasmine)',\n",
        "                 'Trigonella Foenum-graecum (Fenugreek)']\n",
        "    string=\"OUTPUT : \"+class_names[np.argmax(prediction)]\n",
        "    st.success(string)"
      ]
    }
  ]
}
