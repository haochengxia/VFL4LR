for project_name in inst malloc rbt
    do
        pip3 install -U libsvm-official
        bash ./data/adult/get_data.sh
        python3 example.py
    done