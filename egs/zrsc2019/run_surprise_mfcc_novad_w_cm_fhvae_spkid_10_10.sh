#!/bin/bash 

# Copyright 2017 Wei-Ning Hsu
# Apache 2.0 

# zrsc2019_DIR=../../kaldi/egs/zrsc2019/s5
# zrsc2019_RAW_DATA=/lan/ibdata/SPEECH_DATABASE/zrsc2019/
# lang=english
dur=1s
gpu=2
feat_type=mfcc_cm
num_layers=2
n_patience=20
# input sections
len_shift=_10_10
exp_dir=exp_surprise/fhvae_lstm${num_layers}L256_lat32_32_ad10_${feat_type}_novad${len_shift}_spkid/
train_conf=conf/train/fhvae/e200_p${n_patience}_lr1e-3_bs256_nbs2000_ad10.cfg
model_conf=conf/model/fhvae/lstm_${num_layers}L_256_lat_32_32.cfg
feat_dir=data-mfcc_novad
lang=surprise # or surprise
dataset_key=dataset_w_cm${len_shift}
dataset_conf=$feat_dir/$lang/train/unit/${dataset_key}.cfg # we use _english_ as we only use conf for train&dev configuration. _$lang_ only differs in test config.
set_name=spk

# feat_dir=data_merged3lang

tr=train
dt=dev
# tt=test


stage=12

. ./path.sh 
. parse_options.sh || exit 1;

if [ $# -ne 0 ]; then
    echo "Usage: $0: [options]"
    exit 1;
fi

set -eu

# if [ $stage -le -1 ]; then
    # ./local/${feat_type}_data_prep.sh --zrsc2019_RAW_DATA $zrsc2019_RAW_DATA \
        # --zrsc2019_KALDI_EGS $zrsc2019_DIR --KALDI_ROOT $KALDI_DIR || exit 1;
# fi







tr_utt2uttid=$feat_dir/$lang/$tr/unit/utt2uttid
dt_utt2uttid=$feat_dir/$lang/$dt/unit/utt2uttid
# tt_utt2uttid=$feat_dir/$tt/$lang/$dur/utt2uttid

# output sections
tr_z1_meanvar_wspec=ark,scp:$exp_dir/eval/z1_meanvar/train.ark,$exp_dir/eval/z1_meanvar/train.scp
dt_z1_meanvar_wspec=ark,scp:$exp_dir/eval/z1_meanvar/dev.ark,$exp_dir/eval/z1_meanvar/dev.scp
# tt_z1_meanvar_wspec=ark:$exp_dir/eval/z1_meanvar/test_${dur}.ark
if [ $stage -le 0 ]; then
    echo "$0: stage 0, start FHVAE training ($(hostname); $(date))"
    gpu_index=$gpu
    runner_path=/home/siyuan/software/FactorizedHierarchicalVAE/src/runners/
    sed -e "s/='[0-9]'/='${gpu_index}'/" ${runner_path}/fhvae_runner.py > $runner_path/fhvae_runner.py.tmp
    mv $runner_path/fhvae_runner.py.tmp $runner_path/fhvae_runner.py
    python src/scripts/run_nips17_fhvae_exp.py \
        --notest \
        --exp_dir=$exp_dir --set_name=$set_name --dataset_conf=$dataset_conf \
        --train_conf=$train_conf --model_conf=$model_conf || exit 1;

    echo "$0: finished FHVAE training ($(date))"
fi

#if [ $stage -le 1 ]; then
#    echo "$0: stage 1, visualize disentanglement ($(hostname); $(date))"
#    python src/scripts/run_nips17_fhvae_exp.py \
#        --exp_dir=$exp_dir --notest --fac \
#        --fac_z1_spec=$fac_label --fac_z2_spec=$fac_label \
#        --fac_wspec=$fac_wspec --fac_img_dir=$fac_img_dir || exit 1;
#fi

if [ $stage -le 2 ]; then
   echo "$0: stage 2, dump latent variables and s-vectors ($(hostname); $(date))"
   # assume we have copied dataset_{1,10,120}s.cfg files in $exp_dir.
   # cp data/train/$lang/dataset_1s.cfg $exp_dir/.
   # cp data/train/$lang/dataset_10s.cfg $exp_dir/.
   # cp data/train/$lang/dataset_120s.cfg $exp_dir/.
   # rm -f dataset.cfg
   # cp $exp_dir/dataset_1s.cfg $exp_dir/dataset.cfg
   
   # dump on train, dev sets
    gpu_index=$gpu
    runner_path=/home/siyuan/software/FactorizedHierarchicalVAE/src/runners/
    sed -e "s/='[0-9]'/='${gpu_index}'/" ${runner_path}/fhvae_runner.py > $runner_path/fhvae_runner.py.tmp
    mv $runner_path/fhvae_runner.py.tmp $runner_path/fhvae_runner.py
    python src/scripts/run_nips17_fhvae_exp.py \
           --exp_dir=$exp_dir --notest \
           --dump_lat --use_mean --use_logvar --dump_z1 \
           --train_lat_wspec=$tr_z1_meanvar_wspec --train_utt_id_map=$tr_utt2uttid \
           --dev_lat_wspec=$dt_z1_meanvar_wspec --dev_utt_id_map=$dt_utt2uttid || exit 1;
		   
fi
if [ $stage -le 3 ]; then		   
		   
   #!!!#############dump voice data ##############
   dataset_conf_voice=$feat_dir/$lang/voice/${dataset_key}_dump_voice.cfg
   cp $dataset_conf_voice $exp_dir/.
   # cp ${feat_dir}/$tr/${dataset_key}_english_{1s,10s,120s}.cfg $exp_dir/.
	   # dump on 1s set	   
       mv $exp_dir/dataset.cfg $exp_dir/dataset.cfg.orig
	   cp $exp_dir/${dataset_key}_dump_voice.cfg $exp_dir/dataset.cfg
	   tt_utt2uttid=$feat_dir/$lang/voice/utt2uttid
	   tt_z1_meanvar_wspec=ark,scp:$exp_dir/eval/z1_meanvar/voice.ark,$exp_dir/eval/z1_meanvar/voice.scp
	   if [ ! -s $exp_dir/eval/z1_meanvar/$lang/voice.ark ]; then
            gpu_index=$gpu
            runner_path=/home/siyuan/software/FactorizedHierarchicalVAE/src/runners/
            sed -e "s/='[0-9]'/='${gpu_index}'/" ${runner_path}/fhvae_runner.py > $runner_path/fhvae_runner.py.tmp
            mv $runner_path/fhvae_runner.py.tmp $runner_path/fhvae_runner.py
            python src/scripts/run_nips17_fhvae_exp.py \
               --exp_dir=$exp_dir --notest \
               --dump_lat --use_mean --use_logvar --dump_z1 \
               --test_lat_wspec=$tt_z1_meanvar_wspec --test_utt_id_map=$tt_utt2uttid || exit 1;
	   fi
       mv $exp_dir/dataset.cfg.orig $exp_dir/dataset.cfg
fi


if [ $stage -le 4 ]; then		   
		   
   #!!!#############dump test data ##############
   dataset_conf_test=$feat_dir/$lang/test/${dataset_key}_dump_test.cfg
   cp $dataset_conf_test $exp_dir/.
   # cp ${feat_dir}/$tr/${dataset_key}_english_{1s,10s,120s}.cfg $exp_dir/.
	   # dump on 1s set	   
       mv $exp_dir/dataset.cfg $exp_dir/dataset.cfg.orig
	   cp $exp_dir/${dataset_key}_dump_test.cfg $exp_dir/dataset.cfg
	   tt_utt2uttid=$feat_dir/$lang/test/utt2uttid
	   tt_z1_meanvar_wspec=ark,scp:$exp_dir/eval/z1_meanvar/test.ark,$exp_dir/eval/z1_meanvar/test.scp
	   if [ ! -s $exp_dir/eval/z1_meanvar/$lang/test.ark ]; then
            gpu_index=$gpu
            runner_path=/home/siyuan/software/FactorizedHierarchicalVAE/src/runners/
            sed -e "s/='[0-9]'/='${gpu_index}'/" ${runner_path}/fhvae_runner.py > $runner_path/fhvae_runner.py.tmp
            mv $runner_path/fhvae_runner.py.tmp $runner_path/fhvae_runner.py		   
            python src/scripts/run_nips17_fhvae_exp.py \
               --exp_dir=$exp_dir --notest \
               --dump_lat --use_mean --use_logvar --dump_z1 \
               --test_lat_wspec=$tt_z1_meanvar_wspec --test_utt_id_map=$tt_utt2uttid || exit 1;
	   fi
       mv $exp_dir/dataset.cfg.orig $exp_dir/dataset.cfg
fi
# exp2_dir=/lan/ibdata/siyuan/zerospeech2017-master/fhvae_stuff/merged3lang/fhvae_lstm${num_layers}L256_lat32_32_ad10_${feat_type}${len_shift}_spkid/
# if [ $stage -le 4 ]; then
# # <bnf-node-name> <input-datadir> <bnf-data-dir> <nnet-dir>
# # --nj, --use-gpu (default=true)
	# for lang in english french mandarin; do
		# for dur in 1s 10s 120s; do
			# if [ ! -f $exp2_dir/eval/z1_meanvar/$lang/$dur/data_by_utt/0.fea ] && [ ! -f $exp2_dir/eval/z1_meanvar/$lang/$dur/data_by_utt/0.feat ]; then
				# mkdir -p $exp2_dir/eval/z1_meanvar/$lang/$dur/data_by_utt/ || exit 1;
				# copy-feats scp:$exp_dir/eval/z1_meanvar/$lang/test_${dur}.scp scp,t:$exp_dir/eval/z1_meanvar/$lang/foo_${dur}.scp || exit 1;
			# fi

		# done
	# done
# fi


# if [ $stage -le 5 ]; then
    # #first compose train dirs: ./eval/z1_meanvar/train_zrsc2019/{english,french,mandarin}
    # # To do: utils/copy_data_dir.sh /home/siyuan/software/FactorizedHierarchicalVAE/egs/zrsc2019/exp/fhvae_lstm2L256_lat32_32_ad10_mfcc_raw_20_20_spkid/eval/z1_meanvar/train_zrsc2019/{english,french,mandarin} here, and remove {feats,cmvn}.scp
    # # cat eval/z1/meanvar/{train,dev}.scp to each $lang/feats.scp
    # # utils/fix_data_dir.sh to each $lang/
    # original_dir=/home/siyuan/software/FactorizedHierarchicalVAE/egs/zrsc2019/exp/fhvae_lstm2L256_lat32_32_ad10_mfcc_raw_20_20_spkid/eval/z1_meanvar/train_zrsc2019
    # mkdir -p $exp_dir/eval/z1_meanvar/train_zrsc2019/ || exit 1;
    # for lang in english french mandarin; do
        # tgt_dir=$exp_dir/eval/z1_meanvar/train_zrsc2019/$lang
        # ./utils/copy_data_dir.sh $original_dir/$lang $tgt_dir || exit 1;
        # rm -f $tgt_dir/{feats,cmvn}.scp
        # cat $exp_dir/eval/z1_meanvar/train.scp $exp_dir/eval/z1_meanvar/dev.scp >  $tgt_dir/feats.scp
        # ./utils/fix_data_dir.sh $tgt_dir || exit 1;
    # done
# fi


# if [ $stage -le 6 ]; then
    # original_dir=/home/siyuan/software/FactorizedHierarchicalVAE/egs/zrsc2019/exp/fhvae_lstm2L256_lat32_32_ad10_mfcc_raw_20_20_spkid/eval/z1_meanvar/test
    # mkdir -p $exp_dir/eval/z1_meanvar/test/ || exit 1;
    # for lang in english french mandarin; do
        # for dur in 1s 10s 120s; do
            # tgt_dir=$exp_dir/eval/z1_meanvar/test/$lang/$dur
            # ./utils/copy_data_dir.sh $original_dir/$lang/$dur $tgt_dir || exit 1;
            # rm -f $tgt_dir/{feats,cmvn}.scp
            # cp $exp_dir/eval/z1_meanvar/$lang/test_${dur}.scp $tgt_dir/feats.scp
            # ./utils/fix_data_dir.sh $tgt_dir || exit 1;
        # done
    # done
# fi
# src/runners/fhvae_runner.py says in dump_repr process, repr_set_name must be specified.
# while in dump_latent process, *_set_name is not an input variable
tr_repr_wspec=ark,scp:$exp_dir/eval/repr_uttid/train.ark,$exp_dir/eval/repr_uttid/train.scp
dt_repr_wspec=ark,scp:$exp_dir/eval/repr_uttid/dev.ark,$exp_dir/eval/repr_uttid/dev.scp
if [ $stage -le 7 ]; then
    # dump s-vector, mu2, set name=uttid
    dump_train_dev=false
    dump_voice_test=false
    if $dump_train_dev ; then
        python src/scripts/run_nips17_fhvae_exp.py \
                --exp_dir=$exp_dir --notest \
                --dump_repr --repr_set_name=uttid \
                --train_repr_wspec=$tr_repr_wspec --train_repr_id_map=$tr_utt2uttid \
                --dev_repr_wspec=$dt_repr_wspec --dev_repr_id_map=$dt_utt2uttid || exit 1;
    fi
    if $dump_voice_test ; then
        # mkdir -p $exp_dir/eval/repr_uttid/$lang/ || exit 1;
        mv $exp_dir/dataset.cfg $exp_dir/dataset.cfg.orig
        for cate in voice test; do
            # mkdir -p $exp_dir/eval/repr_uttid
            # cp $exp_dir/${dataset_key}_dump_voice.cfg $exp_dir/dataset.cfg
            cp $exp_dir/${dataset_key}_dump_${cate}.cfg $exp_dir/dataset.cfg
            test_repr_wspec=ark,scp:$exp_dir/eval/repr_uttid/${cate}.ark,$exp_dir/eval/repr_uttid/${cate}.scp
            test_utt2uttid=$feat_dir/$lang/$cate/utt2uttid # data/$tt/$lang/${dur}/utt2uttid

            python src/scripts/run_nips17_fhvae_exp.py \
                    --exp_dir=$exp_dir --notest \
                    --dump_repr --repr_set_name=uttid \
                    --test_repr_wspec=$test_repr_wspec --test_repr_id_map=$test_utt2uttid || exit 1;                

        done
        mv $exp_dir/dataset.cfg.orig $exp_dir/dataset.cfg
        
    fi
    dump_train_dev_spk=false
    if $dump_train_dev_spk ; then
        tr_repr_spk_wspec=ark,scp:$exp_dir/eval/repr_spk/train.ark,$exp_dir/eval/repr_spk/train.scp
        dt_repr_spk_wspec=ark,scp:$exp_dir/eval/repr_spk/dev.ark,$exp_dir/eval/repr_spk/dev.scp
        # dump s-vector, mu2, set name=spk
        tr_utt2spkid=$feat_dir/$lang/$tr/unit/utt2spkid
        dt_utt2spkid=$feat_dir/$lang/$dt/unit/utt2spkid
        python src/scripts/run_nips17_fhvae_exp.py \
                --exp_dir=$exp_dir --notest \
                --dump_repr --repr_set_name=spk \
                --train_repr_wspec=$tr_repr_spk_wspec --train_repr_id_map=$tr_utt2spkid \
                --dev_repr_wspec=$dt_repr_spk_wspec --dev_repr_id_map=$dt_utt2spkid || exit 1;
        
    fi

fi


# #!!!!!!!!!!!!!! below are processing related to replaceing utt representation




if [ $stage -le 10 ]; then
# reduplicated from above but now target_repr is S015_4279698902 in english training set
    target_repr_name=S002 # S002 male, S004 Female
    repl_dest_dir=repl_utt_to_${target_repr_name}
    repl_train=false
    repl_dev=false
    repl_test=false
    if $repl_train ; then
        #(be careful! during training CMN) tr_feat_rspec=scp:$feat_dir/$tr/feats.scp
        tr_feat_rspec="ark:apply-cmvn --norm-means=true --norm-vars=false --utt2spk=ark:${feat_dir}/$lang/train/unit/utt2spk scp:${feat_dir}/$lang/train/unit/cmvn.scp scp:${feat_dir}/$lang/train/unit//feats.scp ark:- |"
        # list=repl_conf/repl.txt
        list=repl_conf/repl_${lang}_train_to_${target_repr_name}.txt
        tr_repl_utt_wspec=ark,scp:$exp_dir/eval/$repl_dest_dir/train.ark,$exp_dir/eval/$repl_dest_dir/train.scp
        # tr_repl_utt_wspec=ark:$exp_dir/eval/repl_utt/train.ark
        #--feat_rpec is used to replace test_feat_rspec. Four --feat_* options occur altogether
        tr_repl_utt_img_dir=$exp_dir/eval/$repl_dest_dir/img
        # we manually add a line S015.wav xxx to scp:xxx/train.scp but ark is elsewhere so cannot use ark:xxx/train.ark
        # tr_repr_wspec2=ark:$exp_dir/eval/repr_uttid/train.ark
        tr_repr_wspec2=scp:$exp_dir/eval/repr_uttid/train.scp
        gpu_index=$gpu
        runner_path=/home/siyuan/software/FactorizedHierarchicalVAE/src/runners/
        sed -e "s/='[0-9]'/='${gpu_index}'/" ${runner_path}/fhvae_runner.py > $runner_path/fhvae_runner.py.tmp
        if [ ! -s $exp_dir/eval/$repl_dest_dir/train.ark ]; then
            python src/scripts/run_nips17_fhvae_exp.py \
            --exp_dir=$exp_dir --notest \
            --feat_rspec="$tr_feat_rspec" --feat_set_name=uttid \
            --feat_label_N=$(($(wc -l $tr_utt2uttid | awk '{print $1}') + 1)) \
            --feat_utt2label_path=$tr_utt2uttid \
            --repl_repr_utt --repl_utt_set_name=uttid \
            --repl_utt_repr_spec=$tr_repr_wspec2 --repl_utt_list=$list \
            --repl_utt_id_map=$tr_utt2uttid --repl_utt_wspec=$tr_repl_utt_wspec \
            --repl_utt_img_dir=$tr_repl_utt_img_dir || exit 1;
        fi
    fi
    if $repl_dev ; then
        #(be careful! during training CMN) tr_feat_rspec=scp:$feat_dir/$tr/feats.scp
        feat_rspec="ark:apply-cmvn --norm-means=true --norm-vars=false --utt2spk=ark:${feat_dir}/$lang/dev/unit/utt2spk scp:${feat_dir}/$lang/dev/unit/cmvn.scp scp:${feat_dir}/$lang/dev/unit//feats.scp ark:- |"
        list=repl_conf/repl_${lang}_dev_to_${target_repr_name}.txt
        dt_repl_utt_wspec=ark,scp:$exp_dir/eval/$repl_dest_dir/dev.ark,$exp_dir/eval/$repl_dest_dir/dev.scp
        dt_repl_utt_img_dir=$exp_dir/eval/$repl_dest_dir/img
        # we manually add a line S015.wav xxx to scp:xxx/train.scp but ark is elsewhere so cannot use ark:xxx/train.ark
        # tr_repr_wspec2=ark:$exp_dir/eval/repr_uttid/train.ark
        dt_repr_wspec2=scp:$exp_dir/eval/repr_uttid/dev.scp
        gpu_index=$gpu
        runner_path=/home/siyuan/software/FactorizedHierarchicalVAE/src/runners/
        sed -e "s/='[0-9]'/='${gpu_index}'/" ${runner_path}/fhvae_runner.py > $runner_path/fhvae_runner.py.tmp
        mv $runner_path/fhvae_runner.py.tmp $runner_path/fhvae_runner.py        
        if [ ! -s $exp_dir/eval/$repl_dest_dir/dev.ark ]; then
            python src/scripts/run_nips17_fhvae_exp.py \
            --exp_dir=$exp_dir --notest \
            --feat_rspec="$feat_rspec" --feat_set_name=uttid \
            --feat_label_N=$(($(wc -l $dt_utt2uttid | awk '{print $1}') + 1)) \
            --feat_utt2label_path=$dt_utt2uttid \
            --repl_repr_utt --repl_utt_set_name=uttid \
            --repl_utt_repr_spec=$dt_repr_wspec2 --repl_utt_list=$list \
            --repl_utt_id_map=$dt_utt2uttid --repl_utt_wspec=$dt_repl_utt_wspec \
            --repl_utt_img_dir=$dt_repl_utt_img_dir || exit 1;
        fi
    fi    
    # flag=intra_dur_svec;
    # if $repl_test ; then
        # # test_feat_rspec=scp:$feat_dir/$tt/feats.scp
        # for lang in english french mandarin; do
            # mkdir -p $exp_dir/eval/$repl_dest_dir/$lang/ || exit 1;
            # mkdir -p $exp_dir/eval/$repl_dest_dir/$lang/img || exit 1;
            # for dur in 1s 10s 120s; do
                # # test_utt2uttid=data/$tt/$lang/1s/utt2uttid
                # test_utt2uttid=data/$tt/$lang/${dur}/utt2uttid
                # # list=repl_conf/repl_${lang}_${dur}.txt
                # list=repl_conf/repl_${lang}_${dur}_${target_repr_name}.txt
                # if [ ! -f $list ]; then
                    # exit 1;
                # fi
                # # flag=intra_dur_svec;
                # if [ ! -f $exp_dir/eval/$repl_dest_dir/${lang}/test_${dur}.ark ]; then
                    # # test_feat_rspec=scp:$feat_dir/$tt/$lang/$dur/feats.scp
                    # test_feat_rspec="ark:apply-cmvn --norm-means=true --norm-vars=false --utt2spk=ark:${feat_dir}/test/$lang/$dur/utt2spk scp:${feat_dir}/test/$lang/$dur/cmvn.scp scp:${feat_dir}/test/$lang/$dur/feats.scp ark:- |"
                    
                    # test_repl_utt_wspec=ark,scp:$exp_dir/eval/$repl_dest_dir/${lang}/test_${dur}.ark,$exp_dir/eval/$repl_dest_dir/${lang}/test_${dur}.scp
                    # test_repl_utt_img_dir=$exp_dir/eval/$repl_dest_dir/${lang}/img
                    # # test_repr_wspec2=ark:$exp_dir/eval/repr_uttid/$lang/test_${dur}.ark
                    # test_repr_wspec2=scp:$exp_dir/eval/repr_uttid/$lang/test_${dur}.scp
                    # python src/scripts/run_nips17_fhvae_exp.py \
                    # --exp_dir=$exp_dir --notest \
                    # --feat_rspec="$test_feat_rspec" --feat_set_name=uttid \
                    # --feat_label_N=$(($(wc -l $test_utt2uttid | awk '{print $1}') + 1)) \
                    # --feat_utt2label_path=$test_utt2uttid \
                    # --repl_repr_utt --repl_utt_set_name=uttid \
                    # --repl_utt_repr_spec=$test_repr_wspec2 --repl_utt_list=$list \
                    # --repl_utt_id_map=$test_utt2uttid --repl_utt_wspec=$test_repl_utt_wspec \
                    # --repl_utt_img_dir=$test_repl_utt_img_dir || exit 1;
                # fi
            # done
        # done
        
    # fi
    # next compose data dirs
    s_vec_by=_${target_repr_name}
    compose_data_dir_train=false
    if $compose_data_dir_train ; then
        original_dir=/home/siyuan/software/FactorizedHierarchicalVAE/egs/zrsc2019/data-mfcc_novad/$lang/train_dev/unit
        mkdir -p $exp_dir/eval/$repl_dest_dir/train_zrsc2019/ || exit 1;
        suffix=_modto${s_vec_by} # modto_S015
        echo "$suffix" > $exp_dir/eval/$repl_dest_dir/train_zrsc2019/suffix.txt || exit ; #for future check
        # for lang in english french mandarin; do
        tgt_dir=$exp_dir/eval/$repl_dest_dir/train_zrsc2019/unit
        ./utils/copy_data_dir.sh $original_dir/ $tgt_dir || exit 1;
        rm -f $tgt_dir/{feats,cmvn}.scp
        cat $exp_dir/eval/$repl_dest_dir/train.scp $exp_dir/eval/$repl_dest_dir/dev.scp >  $tgt_dir/feats_w_suffix.scp
        sed -e "s/${suffix}//g" $tgt_dir/feats_w_suffix.scp > $tgt_dir/feats.scp
        ./utils/fix_data_dir.sh $tgt_dir || exit 1;
        # done     
    fi
  
    
    # compose_data_dir_test=false
    # if $compose_data_dir_test ; then
        # original_dir=/home/siyuan/software/FactorizedHierarchicalVAE/egs/zrsc2019/exp/fhvae_lstm2L256_lat32_32_ad10_mfcc_raw_20_20_spkid/eval/z1_meanvar/test
        # mkdir -p $exp_dir/eval/$repl_dest_dir/test/ || exit 1;
        # # s_vec_by=_S015.wav
        # suffix=_modto${s_vec_by}
        # echo "$suffix" > $exp_dir/eval/$repl_dest_dir/test/suffix.txt || exit ; #for future check
        # for lang in english french mandarin; do
            # for dur in 1s 10s 120s; do
                # tgt_dir=$exp_dir/eval/$repl_dest_dir/test/$lang/$dur
                # ./utils/copy_data_dir.sh $original_dir/$lang/$dur $tgt_dir || exit 1;
                # rm -f $tgt_dir/{feats,cmvn}.scp
                # cp $exp_dir/eval/$repl_dest_dir/${lang}/test_${dur}.scp $tgt_dir/feats_w_suffix.scp
                # # we have to remove suffix like 1.wav_modto_0.wav -> 1.wav
                # sed -e "s/${suffix}/g" $tgt_dir/feats_w_suffix.scp > $tgt_dir/feats.scp
                # ./utils/fix_data_dir.sh $tgt_dir || exit 1;
                
            # done
        # done
    # fi
    do_foo_train=false
    if $do_foo_train ; then
        dest_dir=/lan/ibdata/siyuan/zerospeech2019/fhvae_stuff/exp_$lang/fhvae_lstm${num_layers}L256_lat32_32_ad10_${feat_type}_novad${len_shift}_spkid/eval/${repl_dest_dir}/train_zrsc2019/unit
        mkdir -p $dest_dir/data_by_utt/ || exit 1;
        this_data_dir=$exp_dir/eval/$repl_dest_dir/train_zrsc2019/unit
        if [ ! -f $dest_dir/data_by_utt/0.fea ] && [ ! -f $dest_dir/data_by_utt/0.fea2 ]; then
            cut -d " " -f 1 $this_data_dir/utt2spk > $this_data_dir/utt
            sed -e "s/\(.*\)/\1 \/lan\/ibdata\/siyuan\/zerospeech2019\/fhvae_stuff\/exp_${lang}\/fhvae_lstm${num_layers}L256_lat32_32_ad10_${feat_type}_novad${len_shift}_spkid\/eval\/${repl_dest_dir}\/train_zrsc2019\/unit\/data_by_utt\/\1\.fea/g" $this_data_dir/utt > $this_data_dir/foo.scp
            copy-feats scp:$this_data_dir/feats.scp ark:- | add-deltas ark:- scp,t:$this_data_dir/foo.scp || exit 1;
        fi            
        # stamp_dir=/lan/ibdata/siyuan/zerospeech2017-master
        dir=$dest_dir/data_by_utt/
        for file in $dir/*.fea ; do
            sed -e "/\[/d" -e "s/\]//g" $file  > ${file}2;
        done
        # mkdir -p $dir/fea_files;
        # rm -f $dir/*.fea || exit 1;
        find $dir/ -name "*.fea" -delete
    fi    
    
    # # use foo*.scp to generate files for abx evaluation
    # use_foo=false
    # exp2_dir=/lan/ibdata/siyuan/zerospeech2017-master/fhvae_stuff/merged3lang/fhvae_lstm${num_layers}L256_lat32_32_ad10_${feat_type}${len_shift}_spkid/
    # if $use_foo ; then
        # for lang in english french mandarin; do
            # for dur in 1s 10s 120s; do
                # if [ ! -f $exp2_dir/eval/$repl_dest_dir/${lang}/$dur/data_by_utt/0.fea ] && [ ! -f $exp2_dir/eval/$repl_dest_dir/${lang}/$dur/data_by_utt/0.feat ]; then
                    # mkdir -p $exp2_dir/eval/$repl_dest_dir/${lang}/$dur/data_by_utt/ || exit 1;
                    # copy-feats scp:$exp_dir/eval/$repl_dest_dir/test/$lang/$dur/feats.scp scp,t:$exp_dir/eval/$repl_dest_dir/test/$lang/foo_${dur}.scp || exit 1;
                # fi

            # done
        # done
    # fi
fi

# if [ $stage -le 11 ]; then
# #try replace utt representation # try with set_name=spk
    # tr_feat_rspec=scp:$feat_dir/$tr/feats.scp
    # list=repl_conf/repl.txt
    # tr_repl_utt_wspec=ark,scp:$exp_dir/eval/repl_spk/train.ark,$exp_dir/eval/repl_spk/train.scp
    # # tr_repl_utt_wspec=ark:$exp_dir/eval/repl_utt/train.ark
    # tr_repl_utt_img_dir=$exp_dir/eval/repl_spk/img
    # tr_repr_wspec2=ark:$exp_dir/eval/repr_spk/train.ark
    # tr_utt2spkid=$feat_dir/$tr/utt2spkid
    # dt_utt2spkid=$feat_dir/$dt/utt2spkid
    
    # python src/scripts/run_nips17_fhvae_exp.py \
    # --exp_dir=$exp_dir --notest \
    # --feat_rspec=$tr_feat_rspec --feat_set_name=spk \
    # --feat_label_N=$(($(wc -l $tr_utt2spkid | awk '{print $1}') + 1)) \
    # --feat_utt2label_path=$tr_utt2spkid \
    # --repl_repr_utt --repl_utt_set_name=spk \
    # --repl_utt_repr_spec=$tr_repr_wspec2 --repl_utt_list=$list \
    # --repl_utt_id_map=$tr_utt2uttid --repl_utt_wspec=$tr_repl_utt_wspec \
    # --repl_utt_img_dir=$tr_repl_utt_img_dir || exit 1;
# fi
#!!!!!!!!!!!!!! above are processing related to replaceing utt representation


echo "succeeded..."
