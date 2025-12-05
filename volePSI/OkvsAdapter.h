#pragma once

#include "volePSI/Defines.h"
#include "volePSI/Paxos.h"

#include "okvs/encoder.h"

#include <array>
#include <cstring>
#include <memory>
#include <type_traits>
#include <span>
#include <vector>

namespace volePSI
{
	static_assert(sizeof(block) == sizeof(okvs::GF128), "Block type must be 128 bits.");
	static_assert(alignof(block) >= alignof(okvs::GF128), "Block alignment must dominate GF128 alignment.");
	class OkvsAdapter
	{
	public:
		block mSeed = oc::ZeroBlock;
		bool mDebug = false;
		bool mAddToDecode = false;

		void init(u64 numItems, u64 /*binSize*/, u64 weight, u64 ssp, PaxosParam::DenseType /*dt*/, block seed)
		{
			mSeed = seed;
			mNumItems = numItems;

			mConfig.weight = weight;
			mConfig.securityParameter = ssp;

			const auto seed64 = seed.get<u64>(0);
			mEncoder = std::make_unique<okvs::OkvsEncoder>(mConfig, seed64);
			mEncoderSeed = seed;

			mSparseColumns = mEncoder->sparseSize(numItems);
			mDenseColumns = mEncoder->denseSize(numItems);
			mSize = mSparseColumns + mDenseColumns;
		}

		u64 size() const { return mSize; }

		template<typename ValueType>
		void solve(span<const block> inputs,
			span<const ValueType> values,
			span<ValueType> output,
			oc::PRNG* /*prng*/ = nullptr,
			u64 /*numThreads*/ = 0)
		{
			ensureBlockType<ValueType>();

			if (!mEncoder)
				throw RTE_LOC;

			if (inputs.size() != values.size())
				throw RTE_LOC;

			if (output.size() != mSize)
				throw RTE_LOC;

			auto& encoder = ensureEncoder();

			std::vector<okvs::KeyView> keyViews(inputs.size());
			for (std::size_t i = 0; i < inputs.size(); ++i)
				keyViews[i] = keyViewFromBlock(inputs[i]);

			std::vector<okvs::GF128> gfValues(values.size());
			for (std::size_t i = 0; i < values.size(); ++i)
				gfValues[i] = blockToGF128(values[i]);

			auto table = encoder.encode(
				std::span<const okvs::KeyView>(keyViews.data(), keyViews.size()),
				std::span<const okvs::GF128>(gfValues.data(), gfValues.size()));

			if (table.data.size() != output.size())
				throw RTE_LOC;

			std::memcpy(output.data(), table.data.data(), table.data.size() * sizeof(block));
		}

		template<typename ValueType>
		void decode(span<const block> input,
			span<ValueType> values,
			span<const ValueType> table,
			u64 /*numThreads*/ = 0)
		{
			ensureBlockType<ValueType>();

			if (!mEncoder)
				throw RTE_LOC;

			if (input.size() != values.size())
				throw RTE_LOC;

			if (table.size() != mSize)
				throw RTE_LOC;

			auto& encoder = ensureEncoder();

			std::vector<okvs::KeyView> keyViews(input.size());
			for (std::size_t i = 0; i < input.size(); ++i)
				keyViews[i] = keyViewFromBlock(input[i]);

			const auto gfTable = std::span<const okvs::GF128>(
				reinterpret_cast<const okvs::GF128*>(table.data()),
				table.size());

			okvs::OkvsEncoder::EncodedTableView view{
				gfTable,
				mSparseColumns,
				mDenseColumns
			};

			for (std::size_t i = 0; i < input.size(); ++i)
			{
				const auto decoded = encoder.decode(keyViews[i], view);
				values[i] = gf128ToBlock(decoded);
			}
		}

	private:
		okvs::OkvsConfig mConfig;
		u64 mNumItems = 0;
		u64 mSize = 0;
		std::size_t mSparseColumns = 0;
		std::size_t mDenseColumns = 0;
		std::unique_ptr<okvs::OkvsEncoder> mEncoder;
		block mEncoderSeed = oc::ZeroBlock;

		static okvs::KeyView keyViewFromBlock(const block& b)
		{
			return okvs::KeyView{
				reinterpret_cast<const std::uint8_t*>(&b),
				sizeof(block)
			};
		}

		static okvs::GF128 blockToGF128(const block& b)
		{
			std::array<std::uint8_t, 16> bytes{};
			std::memcpy(bytes.data(), &b, bytes.size());
			return okvs::GF128::fromBytes(bytes);
		}

		static block gf128ToBlock(const okvs::GF128& value)
		{
			const auto bytes = value.toBytes();
			block b;
			std::memcpy(&b, bytes.data(), bytes.size());
			return b;
		}

		template<typename T>
		static void ensureBlockType()
		{
			using Decayed = std::remove_cv_t<std::remove_reference_t<T>>;
			static_assert(std::is_same_v<Decayed, block>,
				"OkvsAdapter currently supports ValueType == block");
		}

		okvs::OkvsEncoder& ensureEncoder()
		{
			if (!mEncoder || std::memcmp(&mSeed, &mEncoderSeed, sizeof(block)) != 0)
			{
				mEncoderSeed = mSeed;
				mEncoder = std::make_unique<okvs::OkvsEncoder>(mConfig, mSeed.get<u64>(0));
			}
			return *mEncoder;
		}
	};
}


